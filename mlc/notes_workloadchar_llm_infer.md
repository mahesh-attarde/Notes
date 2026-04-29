# LLM Inference Workload Characterization wrt CPU 

## Overview
+ Collection of very different kernels and runtime phases:

1. **Prefill / prompt processing**
   - Processes many input tokens at once.
   - More GEMM-like.
   - Higher arithmetic intensity.
   - Easier to parallelize across sequence, batch, heads, and matrix tiles.
   - Can benefit from AMX / AVX-512 / cache blocking.

2. **Decode / generation**
   - Processes one new token per request per step.
   - Often small batch.
   - More GEMV / skinny-GEMM-like.
   - Lower arithmetic intensity.
   - More sensitive to runtime overhead, KV-cache bandwidth, synchronization, and scheduling.

3. **Attention**
   - Prefill attention can be compute-heavy.
   - Decode attention is usually KV-cache bandwidth-bound.
   - Long-context decode stresses memory capacity, TLB, LLC, NUMA, and prefetching.

4. **MLP / FFN**
   - Usually the largest dense compute component.
   - Good target for AMX / AVX-512 tiled GEMM.
   - Decode degenerates into skinny GEMM / GEMV.

5. **MoE**
   - Adds sparse, irregular execution.
   - Bottlenecks include routing, top-k, expert imbalance, gather/scatter, and small expert GEMMs.
   - CPU can be competitive because irregular small-batch execution is less hostile to CPU than to GPU.

The key x86 optimization principle is:

> Use static shapes, static memory, contiguous layouts, packed weights, tiled kernels, NUMA-aware placement, and coarse-grained parallelism where possible. Avoid dynamic scheduling, tiny tasks, fragmented KV, and per-token allocation.


# 2. LLM Inference Phases

## 2.1 Prefill

Prefill consumes the input prompt:

```text
input tokens: T_prompt
batch size: B
hidden size: H
layers: L
heads: A
kv heads: A_kv
head dim: D
```

The effective token count is:

```text
M = B * T_prompt
```

Most dense projections look like:

```text
[M x H] * [H x N] -> [M x N]
```

For prefill, `M` can be large, so the computation is GEMM-like.

### Prefill characteristics

| Property | Behavior |
|---|---|
| Arithmetic intensity | Higher |
| Parallelism | Abundant |
| Matrix shape | GEMM-like |
| Cache reuse | Good if tiled |
| Runtime overhead sensitivity | Lower than decode |
| Best backend | AMX / AVX-512 blocked GEMM |
| Bottleneck | Compute, memory bandwidth, attention quadratic term |

### Prefill is good for CPUs when:

- prompt is long,
- batch is moderate,
- weights fit well in cache hierarchy across tiled execution,
- KV can be allocated contiguously,
- GPU baseline must chunk because of VRAM limits.

## 2.2 Decode

Decode generates one token per request per step.

For each step:

```text
M = B_active
```

Often:

```text
B_active = 1..32
```

Dense layers become skinny GEMM or GEMV:

```text
[B x H] * [H x N] -> [B x N]
```

For small `B`, this is memory-bandwidth-sensitive.

### Decode characteristics

| Property | Behavior |
|---|---|
| Arithmetic intensity | Lower |
| Parallelism | Limited by batch/request count |
| Matrix shape | GEMV / skinny GEMM |
| Cache reuse | Harder |
| Runtime overhead sensitivity | Very high |
| KV-cache pressure | High |
| Best backend | AVX-512 GEMV/skinnny GEMM, cached packed weights, operator fusion |
| Bottleneck | Memory bandwidth, KV access, scheduling, synchronization |

### Decode is hard because:

- one token per request gives limited natural parallelism,
- each step touches many weights,
- attention must read the historical KV cache,
- many small operators amplify dispatch overhead,
- batching may be limited by latency SLOs.


# 3. Layer-Level Workload Breakdown

A Transformer layer generally contains:

```text
RMSNorm / LayerNorm
QKV projection
RoPE
Attention
Output projection
Residual add
MLP / FFN
Optional MoE routing + experts
```

## 3.1 RMSNorm / LayerNorm

Shape:

```text
[B*T, H]
```

Operations:

```text
mean or rms reduction
scale
optional residual add
```

### Characteristics

| Property | Behavior |
|---|---|
| Compute | Low |
| Memory traffic | Moderate |
| Arithmetic intensity | Low |
| Parallelism | Across rows and hidden dimension |
| Bottleneck | Memory bandwidth, reduction latency |

### x86 notes

Good implementation should:

- vectorize over hidden dimension,
- use AVX2/AVX-512 reductions,
- fuse residual add + norm if possible,
- avoid separate passes over memory.

Example fusion:

```text
y = rms_norm(x + residual)
```

rather than:

```text
tmp = x + residual
y = rms_norm(tmp)
```


## 3.2 Q/K/V Projection

Operation:

```text
X [M x H] * Wq [H x Hq] -> Q [M x Hq]
X [M x H] * Wk [H x Hkv] -> K [M x Hkv]
X [M x H] * Wv [H x Hkv] -> V [M x Hkv]
```

Often fused as:

```text
X * [Wq | Wk | Wv]
```

or handled as a single QKV operator.

### Characteristics

| Phase | Shape | Bottleneck |
|---|---|---|
| Prefill | GEMM | Compute/cache tiling |
| Decode | GEMV / skinny GEMM | Weight bandwidth |

### x86 notes

Good approaches:

- store weights in layout friendly for output-channel blocking,
- pack weights into `KC x NR` panels,
- use AMX or AVX-512 microkernels,
- fuse Q/K/V into one operator to reuse input activation,
- apply RoPE/RMSNorm to Q/K while output is still hot.


## 3.3 RoPE

RoPE is usually elementwise complex rotation over pairs of hidden elements.

Approximate shape:

```text
[M, num_heads, head_dim]
```

### Characteristics

| Property | Behavior |
|---|---|
| Compute | Low/moderate |
| Memory traffic | Moderate |
| Parallelism | Across tokens, heads, lanes |
| Bottleneck | Memory traffic if separate pass |

### x86 notes

Best treated as a fused post-processing step after Q/K projection.

Instead of:

```text
Q = matmul(...)
Q = rope(Q)
```

prefer:

```text
Q = matmul_and_rope(...)
```


## 3.4 Attention

Attention has very different behavior in prefill and decode.


# 4. Attention Workload Characterization

## 4.1 Prefill Attention

For each layer/head:

```text
Q [T x D]
K [T x D]
V [T x D]
```

Computes:

```text
scores = Q * K^T
probs  = softmax(scores)
out    = probs * V
```

Complexity:

```text
O(B * A * T^2 * D)
```

### Characteristics

| Property | Behavior |
|---|---|
| Compute | High |
| Memory | High |
| Parallelism | Very high |
| Bottleneck | compute, cache reuse, softmax bandwidth |
| Best algorithm | FlashAttention-style blocking |

### x86 parallelization options

- across batch,
- across heads,
- across query blocks,
- across key/value blocks,
- across rows inside softmax,
- across layers if pipelined.

### x86 locality requirement

For good performance, block attention so that:

```text
Q block + K block + V block + partial output
```

fit into cache as much as possible.


## 4.2 Decode Attention

Decode attention for one new token attends to all previous tokens:

```text
Q_new [B, A, D]
K_cache [B, A_kv, T_context, D]
V_cache [B, A_kv, T_context, D]
```

Complexity per generated token:

```text
O(B * A * T_context * D)
```

But it is often memory-bandwidth dominated because it streams KV cache.

### Characteristics

| Property | Behavior |
|---|---|
| Compute | Moderate |
| Memory traffic | Very high |
| Arithmetic intensity | Low |
| Parallelism | Across batch, heads, sequence blocks |
| Bottleneck | KV-cache bandwidth, cache/TLB misses |

### KV-cache bytes

For FP16 KV:

```text
bytes_per_token_per_layer =
  2 tensors * num_kv_heads * head_dim * 2 bytes
```

Example:

```text
num_kv_heads = 8
head_dim = 128
dtype = fp16

bytes/token/layer = 2 * 8 * 128 * 2 = 4096 bytes = 4 KiB
```

For 80 layers and 100k context:

```text
4 KiB * 80 * 100,000 = ~32 GiB KV cache
```

That is just KV, excluding weights.

### x86 implication

Long-context decode is heavily affected by:

- DRAM bandwidth,
- LLC capacity,
- TLB behavior,
- page size,
- NUMA locality,
- KV layout,
- prefetchability.


# 5. MLP / FFN Workload

Typical dense FFN:

```text
gate = X * W_gate
up   = X * W_up
act  = silu(gate) * up
out  = act * W_down
```

For hidden size `H` and intermediate size `I`:

```text
W_gate: [H x I]
W_up:   [H x I]
W_down: [I x H]
```

Usually:

```text
I ~= 3H to 4H
```

### FLOPs per token

Approximate FFN FLOPs:

```text
gate projection: 2 * H * I
up projection:   2 * H * I
down projection: 2 * I * H

total ~= 6 * H * I
```

If `I = 4H`:

```text
total ~= 24 * H^2 FLOPs/token/layer
```

### Characteristics

| Phase | Behavior |
|---|---|
| Prefill | large GEMM, high reuse |
| Decode | skinny GEMM/GEMV, weight bandwidth limited |
| Bottleneck | dense matmul throughput or weight streaming |

### x86 notes

FFN is one of the best places to use:

- AMX,
- AVX-512 FP16/BF16,
- packed weights,
- large pages,
- NUMA-local replication or partitioning,
- fused gate/up kernels.


# 6. MoE Workload

MoE replaces dense FFN with sparse expert execution.

Pipeline:

```text
router logits = X * W_router
top-k experts = topk(router logits)
weights = softmax(top-k logits)

for selected experts:
  expert_out = expert_mlp(X)

output = weighted_sum(expert_out)
```

## 6.1 MoE characteristics

| Component | Bottleneck |
|---|---|
| Router matmul | Small GEMM/GEMV |
| Top-k | branchy / reduction / sort |
| Expert dispatch | irregular |
| Expert MLP | small or medium GEMMs |
| Expert merge | scatter/gather and weighted add |

## 6.2 Why MoE can fit CPUs

MoE is irregular:

- different tokens choose different experts,
- expert load is imbalanced,
- batch per expert may be small,
- gather/scatter overhead is high.

GPUs dislike small irregular batches because SM occupancy and memory coalescing suffer.

CPUs are more tolerant of:

- branchy control flow,
- sparse routing,
- smaller GEMMs,
- pointer-heavy data structures,
- low/moderate parallelism.

## 6.3 MoE parallelization options

| Dimension | Description |
|---|---|
| Token parallelism | Route different tokens independently |
| Expert parallelism | Run different experts in parallel |
| Intra-expert GEMM | Parallelize matrix tiles inside an expert |
| Top-k parallelism | Parallelize router top-k across rows |
| Batch compaction | Group tokens by expert |
| NUMA placement | Place expert weights near worker threads |
| Pipeline | Overlap routing, expert compute, merge |


# 7. Arithmetic Intensity

Arithmetic intensity:

```text
AI = FLOPs / bytes moved
```

High AI means compute-bound.  
Low AI means bandwidth-bound.

## 7.1 Dense GEMM

For:

```text
C[M,N] = A[M,K] * B[K,N]
```

FLOPs:

```text
2 * M * N * K
```

Memory roughly:

```text
bytes = sizeof(dtype) * (M*K + K*N + M*N)
```

Arithmetic intensity improves as `M`, `N`, `K` grow.

Prefill has better AI because `M = B*T` is large.

Decode has worse AI because `M = B` is small.


## 7.2 GEMV / Skinny GEMM

Decode often looks like:

```text
[1 x H] * [H x N]
```

FLOPs:

```text
2 * H * N
```

Bytes:

```text
read weights ~= H * N * dtype_size
read input   ~= H * dtype_size
write output ~= N * dtype_size
```

Arithmetic intensity is approximately:

```text
AI ~= 2 FLOPs / byte for FP16 weights
```

That is bandwidth-sensitive.


## 7.3 Attention Decode

For one token:

```text
Q [D]
K_cache [T,D]
V_cache [T,D]
```

KV bytes dominate:

```text
bytes ~= 2 * T * D * dtype_size
```

FLOPs are roughly:

```text
QK: 2 * T * D
PV: 2 * T * D
total ~= 4 * T * D
```

For FP16:

```text
bytes ~= 2 * T * D * 2 = 4TD
FLOPs ~= 4TD
AI ~= 1 FLOP/byte
```

Very bandwidth-bound.


# 8. x86 Hardware Considerations

## 8.1 SIMD / Matrix Engines

Relevant x86 capabilities:

| Feature | Use |
|---|---|
| AVX2 | baseline vectorization |
| AVX-512F | wide FP32/int vectorization |
| AVX-512 BF16 | BF16 dot-product style acceleration |
| AVX-512 FP16 | native FP16 vector operations |
| AMX-BF16 | high-throughput BF16 tiles |
| AMX-INT8 | quantized inference |
| AMX-FP16 | newer server CPUs, strong FP16 GEMM |

For modern Xeon, the best dense backend is often AMX if available.

## 8.2 Cache hierarchy

LLM inference stresses:

- L1 for microkernel tiles,
- L2 for packed panels,
- LLC for weights/KV/intermediates,
- DRAM for weights and KV streaming.

Key optimization:

```text
make inner loops stream predictable contiguous memory
```

Avoid:

- paged KV pointer chasing,
- scattered expert token layouts,
- random access to weights,
- per-token dynamic allocations.

## 8.3 NUMA

On dual-socket systems, NUMA is often decisive.

Bad case:

```text
thread on socket 0 reads weights/KV allocated on socket 1
```

Good case:

```text
socket-local worker group
socket-local KV partition
socket-local packed weights
first-touch allocation by target worker
```

For long-context workloads, remote DRAM access can destroy scaling.

## 8.4 SMT

SMT/hyperthreading may help bandwidth latency hiding, but can hurt if:

- AVX-512/AMX units are saturated,
- L1/L2 pressure is high,
- memory bandwidth is already saturated.

Recommendation:

- benchmark with SMT on/off,
- often use one thread per physical core for AMX-heavy compute,
- consider SMT for memory-latency-bound decode attention.


# 9. Parallelization Dimensions

LLM inference offers many possible parallelization axes.

## 9.1 Request / batch parallelism

Parallelize across independent requests:

```text
request 0 -> worker group 0
request 1 -> worker group 1
...
```

Best for throughput.

Pros:

- simple,
- little synchronization,
- good cache isolation if weights shared.

Cons:

- hurts latency if batching waits too long,
- duplicate KV/cache working sets,
- can increase memory pressure.


## 9.2 Token / sequence parallelism

During prefill:

```text
different token blocks processed in parallel
```

Good for long prompts.

Parallelize over:

```text
T_prompt blocks
```

Pros:

- abundant parallelism,
- good for prefill,
- enables chunked attention.

Cons:

- attention has dependencies across sequence blocks,
- requires careful KV layout.


## 9.3 Head parallelism

Attention heads are independent before output projection.

Parallelize over:

```text
num_heads
```

Pros:

- natural,
- low synchronization,
- good cache locality if head KV is contiguous.

Cons:

- number of heads may be limited,
- GQA/MQA reduces KV heads,
- per-head work may be too small in decode.


## 9.4 Batch parallelism inside kernels

For matmul:

```text
M = batch * sequence
```

Parallelize over M tiles.

Good when `M` is large.

Prefill:

```text
M large -> good
```

Decode:

```text
M small -> limited
```


## 9.5 Output-channel / N parallelism

For dense projection:

```text
C[M,N]
```

Parallelize over N blocks.

This is common for decode because M is small.

Pros:

- enough work if N is large,
- each worker owns output column blocks,
- B/weight access is contiguous if stored as `N x K`.

Cons:

- all workers read same input vector,
- output reduction not needed, but cache pressure on weights high.


## 9.6 K-reduction parallelism

Split the reduction dimension:

```text
K = K0 + K1 + ...
```

Each worker computes partial sums, then reduce.

Pros:

- increases parallelism for small M/N,
- useful for huge K.

Cons:

- requires reduction of partial C,
- extra memory traffic,
- synchronization overhead.

Use only when M/N parallelism is insufficient.


## 9.7 Layer parallelism

Pipeline different layers across worker groups.

Example:

```text
group 0: layers 0-7
group 1: layers 8-15
...
```

Pros:

- can improve throughput,
- useful if model too large for one socket cache region.

Cons:

- strict layer dependency limits latency improvement,
- pipeline bubbles,
- inter-stage communication.


## 9.8 Tensor parallelism

Split model weights across sockets or nodes.

Example:

```text
column parallel:
  W split by output channels

row parallel:
  W split by input channels
```

On a single x86 server, this can map to NUMA sockets.

Pros:

- reduces per-socket memory capacity requirement,
- increases parallel compute.

Cons:

- requires all-reduce or concat,
- cross-socket communication may dominate.


## 9.9 Expert parallelism

For MoE:

```text
expert 0 -> worker group 0
expert 1 -> worker group 1
...
```

Pros:

- natural sparse parallelism,
- expert weights can be NUMA-local.

Cons:

- load imbalance,
- some experts may receive no tokens,
- routing changes every token.


## 9.10 Vocabulary parallelism

Final logits:

```text
hidden [B,H] * lm_head [V,H] -> logits [B,V]
```

For large vocab `V`, parallelize across vocab blocks.

If only top-k is needed, fuse matmul with top-k:

```text
for vocab tile:
  compute logits tile
  update local top-k
discard logits tile
merge top-k
```

This avoids materializing full logits.


# 10. Recommended Parallelization Strategy by Phase

## 10.1 Prefill

Best axes:

1. batch/token M dimension,
2. output-channel N tiles,
3. heads,
4. attention blocks,
5. layers only if pipelining for throughput,
6. NUMA partitioning for very large models.

Recommended:

```text
Use AMX/AVX512 tiled GEMM.
Use large M blocks.
Use packed weights.
Use FlashAttention-style sequence blocking.
Use NUMA-local allocation.
Use fewer, larger tasks.
```

Avoid:

```text
tiny per-token tasks
barrier after tiny operators
dynamic allocation
paged KV
```


## 10.2 Decode

Best axes:

1. request batching,
2. N/output-channel splitting,
3. head splitting for attention,
4. sequence-block splitting for long-context attention,
5. expert parallelism for MoE,
6. top-k fusion for logits/router.

Recommended:

```text
Keep operator path static.
Fuse small operators.
Use packed weights.
Use GEMV/skinnny-GEMM optimized kernels.
Use contiguous KV.
Use per-thread scratch.
Use top-k fusion.
Avoid full logits materialization where possible.
```

Decode is usually more latency-sensitive than prefill.


## 10.3 Long-context decode

Best axes:

1. attention head parallelism,
2. sequence block parallelism,
3. batch/request parallelism,
4. NUMA partitioning of KV cache.

Recommended KV layout:

```text
[layer][batch/request][kv_head][sequence][head_dim]
```

or a variant where the innermost access for a single attention stream is contiguous.

For a CPU, the key is:

```text
linear scan over KV
predictable prefetch
few TLB misses
no page/block indirection in hot loop
```

Use huge pages if possible.


## 10.4 MoE decode

Best axes:

1. route tokens,
2. compact tokens per expert,
3. parallelize active experts,
4. run intra-expert GEMM,
5. merge results.

Recommended:

```text
Build active expert lists.
Skip inactive experts entirely.
Use per-expert packed weights.
Use expert-local worker groups if expert batch is large.
Use fallback vector kernels if expert batch is tiny.
```


# 11. x86-Specific Kernel Design

## 11.1 Dense GEMM

For FP16/BF16:

Preferred hierarchy:

```text
AMX tile kernel if available
else AVX-512 FP16/BF16
else AVX2 / scalar fallback
```

Blocking:

```text
MC x NC x KC macro tiles
MR x NR micro tiles
```

For AVX-512 FP16:

```text
NR = 32 lanes
```

For AMX:

```text
tile sizes should match AMX tile register dimensions
```

## 11.2 Weight layout

Prefer storing weights in inference-native layout.

For output-channel blocking:

```text
B_nt = [N, K]
```

Then for an output block of `NR` columns:

```text
B_panel = [KC, NR]
```

This gives contiguous vector loads:

```text
load B_panel[k, 0:NR]
```

## 11.3 Activation layout

For matmul:

```text
A = [M, K]
```

Make K contiguous.

For decode, the input vector should be cache-hot and reused across N blocks.

## 11.4 KV layout

For decode attention, optimize for scanning historical tokens.

Possible layout:

```text
K[layer][request][kv_head][seq][head_dim]
V[layer][request][kv_head][seq][head_dim]
```

Access for one head:

```text
for t in sequence:
  load K[t, 0:D]
```

This is prefetch-friendly.

Avoid layouts requiring:

```text
page lookup -> block pointer -> offset -> load
```

inside the innermost attention loop.


# 12. Synchronization Strategy

## 12.1 Good synchronization

Use coarse barriers:

```text
after large operator
after layer
after dependent phase
```

## 12.2 Bad synchronization

Avoid barriers after tiny kernels:

```text
RMSNorm
small add
small activation
tiny routing op
```

unless they are fused or grouped.

## 12.3 Better model

Instead of:

```text
for op:
  run op
  barrier
```

consider:

```text
for fused_superop:
  run many dependent tiny ops locally
  barrier only when cross-thread dependency appears
```


# 13. NUMA Parallelization
For large x86 servers, NUMA matters as much as SIMD.

## 13.1 Socket-level partitioning

Options:

### Replicated weights

Each socket has its own copy of weights.

Pros:

- local reads,
- simple scheduling.

Cons:

- doubles memory use.

### Partitioned weights

Split output channels or layers by socket.

Pros:

- lower memory replication,
- more capacity.

Cons:

- requires cross-socket reduce/concat.

### Partitioned KV

Assign requests to sockets and keep their KV local.

Pros:

- critical for long-context decode,
- avoids remote DRAM.

Cons:

- load balancing across sockets is harder.

## 13.2 First-touch policy

Allocate and initialize memory on the same socket that will use it.

Bad:

```text
main thread allocates all weights on socket 0
all sockets read remotely
```

Good:

```text
worker group on socket i initializes its own shard/copy
```


# 14. Workload Regimes

## 14.1 Short prompt, small batch, decode-heavy

Characteristics:

```text
T_prompt small
B small
output tokens many
```

Bottlenecks:

- GEMV bandwidth,
- runtime overhead,
- KV reads,
- small-op synchronization.

Parallelization:

- request batching,
- output-channel parallelism,
- head parallelism,
- fuse ops aggressively.


## 14.2 Long prompt, prefill-heavy

Characteristics:

```text
T_prompt large
TTFT critical
```

Bottlenecks:

- dense GEMM,
- attention,
- weight/KV movement,
- memory capacity.

Parallelization:

- token/block parallelism,
- GEMM tile parallelism,
- head parallelism,
- attention block parallelism,
- NUMA-aware prefill.

This is the regime where CPU can be surprisingly strong if GPU chunking is severe.


## 14.3 Long-context decode

Characteristics:

```text
T_context very large
one token per step
```

Bottlenecks:

- KV cache bandwidth,
- TLB/cache misses,
- NUMA locality.

Parallelization:

- heads,
- sequence blocks,
- batch/request,
- socket-local KV partition.

Most important optimization:

```text
contiguous KV + prefetch + huge pages + NUMA locality
```


## 14.4 MoE small-batch decode

Characteristics:

```text
small B
random expert activation
sparse expert compute
```

Bottlenecks:

- load imbalance,
- expert dispatch,
- small GEMM inefficiency,
- gather/scatter.

Parallelization:

- active expert parallelism,
- token compaction by expert,
- intra-expert tiles,
- fallback vector kernels for tiny expert batches.


# 15. Steps to optimize
## Priority 1: Eliminate runtime overhead

- static graph / static operator queue,
- no allocation in decode loop,
- no dynamic shape logic in hot path,
- no per-token task queue if avoidable.

## Priority 2: Fix memory layout

- weights in inference-native packed/NT format,
- contiguous KV,
- aligned buffers,
- huge pages,
- NUMA placement.

## Priority 3: Use correct kernels

- AMX for large BF16/FP16 GEMM,
- AVX-512 for small/skinnny/tail kernels,
- specialized GEMV for decode,
- fused top-k kernels.

## Priority 4: Fuse operators

Important fusions:

```text
QKV projection
matmul + bias/residual
RMSNorm + projection
matmul + RoPE
gate + up + SiLU + multiply
matmul + top-k
expert down + weighted add
residual + norm
```

## Priority 5: Parallelize by regime

Do not use one strategy everywhere.

| Regime | Best parallelism |
|---|---|
| Prefill | token/batch + GEMM tiles + heads |
| Decode dense | requests + output channels + heads |
| Decode long-context | heads + sequence blocks + NUMA-local KV |
| MoE | expert + token compaction + intra-expert GEMM |
| Logits/top-k | vocab blocks + local top-k heaps |


## design

```text
1. Pre-convert weights to backend-native layout.
2. Build static operator graph once.
3. Allocate all tensors/KV upfront.
4. Pin worker threads to cores.
5. Partition work by phase:
   - prefill: large GEMM/attention tiles
   - decode: output-channel/head/request parallelism
   - long-context: sequence-block KV parallelism
   - MoE: expert/token parallelism
6. Use NUMA-aware memory placement.
7. Fuse small operators into larger backend kernels.
8. Avoid materializing full outputs when only top-k is needed.
9. Use AMX where profitable, AVX-512 for narrow/tail cases.
10. Measure barrier wait time, bandwidth, LLC misses, and NUMA remote traffic.
```


# 18. Snippets
## 18.1 Dense Projection / Linear Layer

Most LLM linear layers compute:

```text
C[M, N] = A[M, K] * B[K, N]
```

For inference-friendly layout, store weights as:

```text
B_nt[N, K]
```

Then:

```text
C[m, n] = sum_k A[m, k] * B_nt[n, k]
```

```c name=dense_projection.c
#include <stddef.h>

void dense_projection_f32(
    const float *a,     // [M, K]
    const float *b_nt,  // [N, K]
    float *c,           // [M, N]
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;

            for (size_t k = 0; k < K; ++k) {
                acc += a[m * K + k] * b_nt[n * K + k];
            }

            c[m * N + n] = acc;
        }
    }
}
```

### Conceptual blocked form

This mirrors the structure of an optimized x86 kernel:

```text
A tile:  MR x KC
B tile:  KC x NR
C tile:  MR x NR
```

For AVX-512 FP16, a common choice is:

```text
NR = 32 FP16 lanes
```

```c name=blocked_dense_projection.c
#include <stddef.h>

void blocked_dense_projection_f32(
    const float *a,     // [M, K]
    const float *b_nt,  // [N, K]
    float *c,           // [M, N]
    size_t M,
    size_t N,
    size_t K,
    size_t KC
) {
    const size_t MR = 3;
    const size_t NR = 32;

    for (size_t m0 = 0; m0 < M; m0 += MR) {
        for (size_t n0 = 0; n0 < N; n0 += NR) {
            float tile[3][32] = {0};

            for (size_t k0 = 0; k0 < K; k0 += KC) {
                size_t kend = k0 + KC;
                if (kend > K) {
                    kend = K;
                }

                for (size_t k = k0; k < kend; ++k) {
                    for (size_t r = 0; r < MR; ++r) {
                        size_t m = m0 + r;
                        if (m >= M) {
                            continue;
                        }

                        float a_val = a[m * K + k];

                        for (size_t lane = 0; lane < NR; ++lane) {
                            size_t n = n0 + lane;
                            if (n >= N) {
                                continue;
                            }

                            tile[r][lane] += a_val * b_nt[n * K + k];
                        }
                    }
                }
            }

            for (size_t r = 0; r < MR; ++r) {
                size_t m = m0 + r;
                if (m >= M) {
                    continue;
                }

                for (size_t lane = 0; lane < NR; ++lane) {
                    size_t n = n0 + lane;
                    if (n >= N) {
                        continue;
                    }

                    c[m * N + n] = tile[r][lane];
                }
            }
        }
    }
}
```


## 18.2 AVX-512-Style FP32 Microkernel Sketch

This is a simplified AVX-512 FP32 sketch for a `3 x 16` tile.

For FP32:

```text
__m512 = 16 lanes
NR = 16
MR = 3
```

For FP16 AVX-512, the idea is similar but uses 32 FP16 lanes.

```c name=avx512_f32_microkernel_sketch.c
#include <stddef.h>
#include <immintrin.h>

// Computes:
// C[3, 16] += A[3, KC] * B_panel[KC, 16]
//
// A is row-major with leading dimension lda.
// B_panel is packed as [KC, 16] contiguous.
// C is row-major with leading dimension ldc.
void matmul_3x16_f32_avx512(
    const float *a,
    const float *b_panel,
    float *c,
    size_t KC,
    size_t lda,
    size_t ldc
) {
    __m512 c0 = _mm512_loadu_ps(c + 0 * ldc);
    __m512 c1 = _mm512_loadu_ps(c + 1 * ldc);
    __m512 c2 = _mm512_loadu_ps(c + 2 * ldc);

    for (size_t k = 0; k < KC; ++k) {
        __m512 b = _mm512_loadu_ps(b_panel + k * 16);

        __m512 a0 = _mm512_set1_ps(a[0 * lda + k]);
        __m512 a1 = _mm512_set1_ps(a[1 * lda + k]);
        __m512 a2 = _mm512_set1_ps(a[2 * lda + k]);

        c0 = _mm512_fmadd_ps(a0, b, c0);
        c1 = _mm512_fmadd_ps(a1, b, c1);
        c2 = _mm512_fmadd_ps(a2, b, c2);
    }

    _mm512_storeu_ps(c + 0 * ldc, c0);
    _mm512_storeu_ps(c + 1 * ldc, c1);
    _mm512_storeu_ps(c + 2 * ldc, c2);
}
```

Conceptually, the FP16 version becomes:

```text
C[3, 32] += A[3, KC] * B_panel[KC, 32]
```

with:

```text
32 FP16 lanes per AVX-512 vector
```


## 18.3 Decode Attention for One Head

During decode, one new query token attends over all previous tokens.

Shapes:

```text
q       = [D]
K_cache = [T, D]
V_cache = [T, D]
out     = [D]
```

Computation:

```text
score[t] = dot(q, K_cache[t]) * scale
prob[t]  = softmax(score[t])
out      = sum_t prob[t] * V_cache[t]
```

```c name=decode_attention_one_head.c
#include <stddef.h>
#include <float.h>
#include <math.h>

void decode_attention_one_head_f32(
    const float *q,        // [D]
    const float *k_cache,  // [T, D]
    const float *v_cache,  // [T, D]
    float *scores,         // [T], scratch
    float *out,            // [D]
    size_t T,
    size_t D,
    float scale
) {
    // QK dot products
    for (size_t t = 0; t < T; ++t) {
        float acc = 0.0f;

        for (size_t d = 0; d < D; ++d) {
            acc += q[d] * k_cache[t * D + d];
        }

        scores[t] = acc * scale;
    }

    // Softmax max
    float max_score = -FLT_MAX;
    for (size_t t = 0; t < T; ++t) {
        if (scores[t] > max_score) {
            max_score = scores[t];
        }
    }

    // Softmax exp and denominator
    float denom = 0.0f;
    for (size_t t = 0; t < T; ++t) {
        float e = expf(scores[t] - max_score);
        scores[t] = e;
        denom += e;
    }

    float inv_denom = 1.0f / denom;

    for (size_t t = 0; t < T; ++t) {
        scores[t] *= inv_denom;
    }

    // Initialize output
    for (size_t d = 0; d < D; ++d) {
        out[d] = 0.0f;
    }

    // Weighted sum over V
    for (size_t t = 0; t < T; ++t) {
        float p = scores[t];

        for (size_t d = 0; d < D; ++d) {
            out[d] += p * v_cache[t * D + d];
        }
    }
}
```

### Why this is bandwidth-heavy

For each generated token, decode attention streams:

```text
K_cache[0:T, 0:D]
V_cache[0:T, 0:D]
```

For FP16 KV, approximate bytes per head are:

```text
K bytes = T * D * 2
V bytes = T * D * 2
total   = 4 * T * D bytes
```

So for large `T`, performance is often limited by memory bandwidth and cache/TLB behavior.


## 18.4 Fused MatMul + Top-K

This demonstrates the top-k fusion idea used for router logits or final vocabulary logits.

Instead of materializing all logits:

```text
logits[N] = hidden[H] * weight[N, H]
topk(logits)
```

compute one logit at a time and maintain a small top-k buffer.

```c name=matmul_topk_one_row.c
#include <stddef.h>
#include <float.h>

static void insert_topk_desc(
    float value,
    size_t index,
    float *top_values,    // [Ktop], descending
    size_t *top_indices,  // [Ktop]
    size_t Ktop
) {
    if (value <= top_values[Ktop - 1]) {
        return;
    }

    size_t pos = Ktop - 1;

    while (pos > 0 && value > top_values[pos - 1]) {
        top_values[pos] = top_values[pos - 1];
        top_indices[pos] = top_indices[pos - 1];
        --pos;
    }

    top_values[pos] = value;
    top_indices[pos] = index;
}

void matmul_topk_one_row_f32(
    const float *a,        // [H]
    const float *b_nt,     // [N, H]
    float *top_values,     // [Ktop]
    size_t *top_indices,   // [Ktop]
    size_t N,
    size_t H,
    size_t Ktop
) {
    for (size_t i = 0; i < Ktop; ++i) {
        top_values[i] = -FLT_MAX;
        top_indices[i] = 0;
    }

    for (size_t n = 0; n < N; ++n) {
        float acc = 0.0f;

        for (size_t h = 0; h < H; ++h) {
            acc += a[h] * b_nt[n * H + h];
        }

        insert_topk_desc(acc, n, top_values, top_indices, Ktop);
    }
}
```

Parallel version:

```text
thread 0: columns [0, N0), local top-k
thread 1: columns [N0, N1), local top-k
...
final step: merge all local top-k lists
```

This avoids writing the full `N`-element logits vector to memory.


## 18.5 Simplified MoE Routing

A minimal MoE routing step:

```text
logits[e] = dot(x, router_weight[e])
topk experts = topk(logits)
weights = softmax(selected logits)
```

```c name=moe_routing_topk.c
#include <stddef.h>
#include <float.h>
#include <math.h>

static void insert_expert_topk(
    float value,
    size_t expert_id,
    float *top_values,
    size_t *top_indices,
    size_t Ktop
) {
    if (value <= top_values[Ktop - 1]) {
        return;
    }

    size_t pos = Ktop - 1;

    while (pos > 0 && value > top_values[pos - 1]) {
        top_values[pos] = top_values[pos - 1];
        top_indices[pos] = top_indices[pos - 1];
        --pos;
    }

    top_values[pos] = value;
    top_indices[pos] = expert_id;
}

void moe_route_one_token_f32(
    const float *x,           // [H]
    const float *router_w_nt, // [E, H]
    float *top_values,        // [Ktop], output probabilities
    size_t *top_indices,      // [Ktop], expert ids
    size_t E,
    size_t H,
    size_t Ktop
) {
    for (size_t i = 0; i < Ktop; ++i) {
        top_values[i] = -FLT_MAX;
        top_indices[i] = 0;
    }

    // Router matmul + top-k
    for (size_t e = 0; e < E; ++e) {
        float acc = 0.0f;

        for (size_t h = 0; h < H; ++h) {
            acc += x[h] * router_w_nt[e * H + h];
        }

        insert_expert_topk(acc, e, top_values, top_indices, Ktop);
    }

    // Softmax over selected experts
    float max_val = top_values[0];

    float denom = 0.0f;
    for (size_t i = 0; i < Ktop; ++i) {
        float e = expf(top_values[i] - max_val);
        top_values[i] = e;
        denom += e;
    }

    float inv_denom = 1.0f / denom;
    for (size_t i = 0; i < Ktop; ++i) {
        top_values[i] *= inv_denom;
    }
}
```

The full optimized MoE implementation would then:

```text
1. group tokens by selected expert,
2. run selected expert MLPs,
3. scale outputs by routing weights,
4. merge them back into token order.
```


## 18.6 RMSNorm

RMSNorm is common in LLMs.

Formula:

```text
rms = sqrt(mean(x_i^2) + eps)
y_i = x_i / rms * weight_i
```

```c name=rmsnorm.c
#include <stddef.h>
#include <math.h>

void rmsnorm_f32(
    const float *x,       // [H]
    const float *weight,  // [H]
    float *y,             // [H]
    size_t H,
    float eps
) {
    float sum_sq = 0.0f;

    for (size_t i = 0; i < H; ++i) {
        sum_sq += x[i] * x[i];
    }

    float mean_sq = sum_sq / (float)H;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

    for (size_t i = 0; i < H; ++i) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}
```

Fused residual + RMSNorm:

```c name=fused_residual_rmsnorm.c
#include <stddef.h>
#include <math.h>

void fused_residual_rmsnorm_f32(
    const float *x,         // [H]
    const float *residual,  // [H]
    const float *weight,    // [H]
    float *y,               // [H]
    size_t H,
    float eps
) {
    float sum_sq = 0.0f;

    for (size_t i = 0; i < H; ++i) {
        float v = x[i] + residual[i];
        y[i] = v;           // temporary stored in output
        sum_sq += v * v;
    }

    float mean_sq = sum_sq / (float)H;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

    for (size_t i = 0; i < H; ++i) {
        y[i] = y[i] * inv_rms * weight[i];
    }
}
```

This avoids writing a separate residual-add temporary.
