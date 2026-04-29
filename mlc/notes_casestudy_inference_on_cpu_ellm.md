# Ellm: Compiler / Runtime / Kernel Performance Notes

## Outline 
eLLM attacks CPU inference bottlenecks by turning LLM execution into a mostly static, cache-friendly, preallocated, SIMD-oriented pipeline.

The main bottlenecks it tries to remove are:
| Bottleneck | eLLM design response |
|---|---|
| Per-token dynamic scheduling overhead | Static operator queue built once from tensor graph construction |
| Temporary tensor allocation churn | Preallocated cache and reused intermediate buffers |
| Fragmented / paged KV access | Static-shape contiguous KV-like tensors and direct coordinate access |
| Poor cache locality | Dimension-first / contiguous layouts, packed B panels, head-by-head attention design |
| Underutilized CPU SIMD | AVX-512 FP16 microkernels, fixed `MR=3`, `NR=32` tiles |
| Q/K/V projection overhead | Fused `MatMul3` operator for Q, K, V paths |
| Materializing large logits before top-k | `MatMulTopK` computes tiles and immediately inserts into per-thread heaps |
| Thread migration and unstable cache residency | Core affinity pinning in serving runtime |
| General-purpose framework overhead | Lean Rust runtime, raw pointers, fixed dispatch enum |

## 2. Important repository locations

### Runtime / scheduling

- `src/serving/start.rs`  
  Core-pinned worker runtime.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/serving/start.rs

- `src/runtime/barrier.rs`  
  Custom spin barrier with cache-line padded atomics.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/runtime/barrier.rs

- `src/compiler/operator.rs`  
  Static `Operator<T>` enum and dispatch.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/compiler/operator.rs

### Tensor graph construction / memory

- `src/ptensor/tensor.rs`  
  Tensor API, graph/operator construction, tensor cache usage.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/ptensor/tensor.rs

- `src/memory/cache.rs`  
  Buffer reuse policy.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/memory/cache.rs

- `src/memory/allocator.rs`  
  64-byte aligned allocation.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/memory/allocator.rs

### Compiler operators

- `src/compiler/mul/matmul.rs`  
  Packed-B tiled GEMM runner.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/compiler/mul/matmul.rs

- `src/compiler/mul/matmul3.rs`  
  Fused Q/K/V GEMM runner with optional RMSNorm/RoPE finalization.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/compiler/mul/matmul3.rs

- `src/compiler/mul/matmul_topk.rs`  
  GEMM + local top-k without materializing full output.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/compiler/mul/matmul_topk.rs

- `src/compiler/assign.rs`  
  Static tile assignment.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/compiler/assign.rs

### AVX-512 FP16 kernels

- `src/kernel/x86_64/f16_512/matmul_block.rs`  
  Main `3 x 32` FP16 matmul microkernel.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/kernel/x86_64/f16_512/matmul_block.rs

- `src/kernel/x86_64/f16_512/matmul_rms_complex.rs`  
  Q/K finalization path: matmul accumulation + RMSNorm + RoPE style complex operation.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/kernel/x86_64/f16_512/matmul_rms_complex.rs

- `src/kernel/x86_64/f16_512/moe_silu.rs`  
  MoE gate/up fused matmul + SiLU/mul support.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/kernel/x86_64/f16_512/moe_silu.rs

- `src/kernel/x86_64/f16_512/experts_topk_softmax_norm.rs`  
  Expert top-k / softmax specialization.  
  https://github.com/lucienhuangfu/eLLM/blob/5039511a23956d808d3b36b0257c3124710a2be2/src/kernel/x86_64/f16_512/experts_topk_softmax_norm.rs

## 3. Performane
It is not “CPU raw FLOPs beat GPU raw FLOPs” in the usual dense-batch case.
The thesis is narrower:

1. Long-context inference is often prefill-dominated.
2. GPU VRAM limits force chunking for very long contexts.
3. Chunking causes repeated parameter/KV movement, repeated scheduling points, synchronization, and state stitching.
4. CPU servers have much larger DRAM capacity and large shared LLC.
5. If eLLM can process long prefill more continuously, with contiguous KV and less runtime overhead, CPU can win end-to-end latency for specific workloads.

It explicitly identifies the CPU baseline bottlenecks as:
- scheduling overhead,
- KV cache management overhead,
- intermediate tensor management overhead,
- service/runtime overhead.

eLLM’s implementation tries to remove these via:

- static operator queues,
- preallocated buffers,
- contiguous layouts,
- core-pinned workers,
- custom tiled SIMD kernels.

## 4. End-to-end operation lifecycle

## 4.1 Model construction phase

`src/bin/main.rs` constructs a Qwen3-MoE model and calls `model.forward(sequences)`.

Important behavior:

1. `Model::<f16>::new(...)` builds the model structure.
2. `model.forward(sequences)` does not simply execute immediately.
3. Tensor operations append `Operator<T>` values into a shared `operator_queue`.
4. Runtime later consumes this fixed queue.

The runtime entry:

```text
main()
  -> Config::load_from_file(...)
  -> Model::<f16>::new(...)
  -> allocate_init::<usize>(...)
  -> model.forward(sequences)
  -> start(model.operator_queue.take(), sequence_length, batch_size)
```
This is closer to a static graph capture model than eager dynamic execution.


## 4.2 Tensor API as graph builder

The central abstraction is `Tensor<T>` in `src/ptensor/tensor.rs`.

A `Tensor<T>` contains:

```text
data: raw pointer
shape: Vec<usize>
strides: Vec<usize>
tensor_name: String
cache: Rc<RefCell<Cache<T>>>
operator_queue: Rc<RefCell<Vec<Operator<T>>>>
```

Most Tensor methods:

1. allocate or retrieve output tensor from cache,
2. construct an `Operator<T>` enum variant,
3. push it to `operator_queue`,
4. return output tensor metadata.

Examples:

- `Tensor::add()` emits `Operator::AddZipMap`.
- `Tensor::matmul()` emits `Operator::MatMul`.
- `Tensor::matmul3()` emits `Operator::MatMul3`.
- `Tensor::attention()` emits `Operator::Attention`.
- `Tensor::experts_matmul_silu_mul_matmul()` emits `Operator::ExpertsMatMulSilu`.

This removes a large part of dynamic runtime overhead because graph structure is built once, then replayed.

Compiler-performance interpretation:

```text
Tensor method call = graph IR construction
Operator enum       = low-level scheduled IR node
Operator::run      = backend dispatch
kernel::*          = target-specific lowering
```

It is not a sophisticated optimizer yet, but the shape is compiler-like.

## 4.3 Memory allocation and reuse

### 4.3.1 64-byte aligned allocation

The allocator uses `Layout::from_size_align_unchecked(..., 64)`.

Performance implication:

- 64-byte alignment matches common cache-line size.
- It is also friendly for AVX-512 512-bit vectors.
- Even though many kernels use unaligned loads/stores, aligned allocation reduces split-line penalties and improves predictability.

### 4.3.2 Cache-based tensor reuse

`Cache<T>` maps tensor names to allocated raw pointers.

The key behavior:

1. Parameters ending in `weight` allocate parameter-like storage.
2. KV tensors allocate and are stored uniquely.
3. Layer intermediates are canonicalized by stripping `model.layers.N.` prefix.
4. Different layers can reuse intermediate storage with the same suffix.
5. Other outputs are stored separately.

This is a major runtime bottleneck removal mechanism:

| Traditional runtime | eLLM behavior |
|---|---|
| Allocate temporary Q/K/V, MLP activations, residual buffers repeatedly | Allocate once / reuse through cache |
| Dynamic lifetime tracking | Name/regex-based reuse |
| General allocator overhead | Raw pointer cache |
| Fragmentation risk | Long-lived buffers |

This is crude but effective for prototype performance.

Compiler-performance note:

The cache acts like a manually implemented static memory planner. It is not yet a formal liveness allocator, but it serves the same performance purpose: eliminate dynamic allocation in hot inference loops.


## 4.4 Static operator queue

The `Operator<T>` enum in `src/compiler/operator.rs` contains variants such as:

```text
AddRMSZipMap
AddZipMap
Attention
ComplexZipMap
ExpertsMatMulDown
ExpertsMatMulSilu
ExpertsMergeAdd
ExpertsSoftmaxNorm
LookupRMSMap
MatMul
MatMul3
MatMulAdd
MatMulTopK
RMSMap
SiluMulZipMap
TopKSoftmax
```

`Operator::run(...)` dispatches to the concrete operator’s `run(...)`.

Signature pattern:

```text
run(
  position_index,
  position_interval,
  batch_size,
  cpu_num,
  thread_id
)
```

Performance interpretation:

- `Operator` is a fixed IR node.
- `position_index` / `position_interval` represent sequence/decode windowing.
- `cpu_num` and `thread_id` control deterministic work partitioning.
- Every worker runs the same queue, but each operator only processes the tile range assigned to that worker.

## 5. Runtime execution model

`src/serving/start.rs` performs runtime execution.

Core behavior:

1. Determine number of worker threads:
   ```text
   thread_num = available_parallelism()
   ```

2. Wrap static operator queue in `Arc`.

3. Create a custom `Barrier`.

4. Obtain physical/logical core ids using `core_affinity`.

5. Spawn one worker per core.

6. Pin each worker to its core:
   ```text
   core_affinity::set_for_current(core_id)
   ```

7. For every decode step:
   ```text
   for _p in 0..sequence_length {
       for operator in queue.iter() {
           operator.run(0, 1, batch_size, thread_num, thread_id);
           barrier.wait();
       }
   }
   ```

### 5.1 Why this removes CPU bottlenecks

Compared with general-purpose serving frameworks, this design avoids:

- token-level dynamic scheduling,
- request merging/splitting,
- Python/GIL interaction,
- framework queueing overhead,
- kernel launch overhead,
- repeated graph construction,
- most dynamic allocation.

### 5.2 Why the barrier matters

The runtime inserts a barrier after every operator.

That enforces graph dependency order:

```text
all threads finish operator i
  -> all threads proceed to operator i+1
```

The barrier implementation in `src/runtime/barrier.rs` uses:

- `AtomicUsize arrived`,
- `AtomicUsize generation`,
- `#[repr(align(64))]` cache padding,
- spin-wait loop with `std::hint::spin_loop()`.

This is low-latency for short synchronized phases but has tradeoffs:

| Strength | Risk |
|---|---|
| Avoids OS blocking/wakeup overhead | Burns CPU while waiting |
| Low overhead for tight compute phases | Bad if operator load imbalance exists |
| Cache-line padding avoids false sharing between counters | Still centralized synchronization |
| Simple deterministic execution | Barrier after every operator may become dominant for tiny operators |

For compiler/runtime tuning, barrier frequency is an important performance knob.

## 6. Work partitioning

`src/compiler/assign.rs` provides simple deterministic partitioning.

Given:

```text
length = total tiles
num    = cpu_num
id     = thread_id
```

It returns a contiguous tile interval:

```text
[begin, end)
```

Remainder is distributed to low thread ids.

Performance properties:

- deterministic,
- no runtime task queue,
- no atomic work stealing,
- no dynamic scheduling overhead,
- good for uniform tiles.

Potential downside:

- no load balancing if tiles have nonuniform cost,
- bad for sparse MoE imbalance unless higher-level operators compensate.

There is also `assign_kqv_tile(...)` for distributing V/K/Q tile groups, but current `MatMul3::run()` uses a flattened task list plus `assign(total_tasks, cpu_num, thread_id)`.

# 7. GEMM operator: `MatMul<T>`

File: `src/compiler/mul/matmul.rs`

## 7.1 Data layout contract

`MatMul<T>` uses:

```text
A      = [M x K]
B_nt   = [N x K] row-major
C      = [M x N]
```

This is important.

Instead of taking B in normal `[K x N]` layout and transposing internally each time, the code assumes weights are already in NT layout:

```text
B_nt[j, k] = logical B[k, j]
```

Performance reason:

- for a fixed output column block, each B row is contiguous over K,
- B panels can be packed in microkernel-friendly form,
- avoids repeated transpose cost.

## 7.2 Constructor phase

`MatMul::new(...)`:

1. receives raw pointers to `A`, `B_nt`, `C`,
2. stores shape maxima:
   ```text
   m_max, n_max, k_max
   ```
3. stores tiling parameters:
   ```text
   MB = a_row_step_macro
   NB = b_row_step_macro
   KC = column_step_macro
   MR = a_row_step_micro
   NR = b_row_step_micro
   ```
4. pre-packs B into panel layout.

### 7.2.1 B panel packing

Packing layout:

```text
packed_b = [panels_k][panels_n][KC * NR]
```

For each K panel and N panel:

```text
for p in 0..KC:
    for lane in 0..NR:
        packed[p * NR + lane] = B_nt[n0 + lane, k0 + p]
```

This produces the exact layout consumed by the AVX-512 microkernel:

```text
for k in 0..KC:
    bvec = load 32 contiguous f16 values
```

### 7.2.2 Packed panel address

Panel pointer computation:

```text
panels_n = ceil(N / NR)
panel_idx = (k0 / KC) * panels_n + (n0 / NR)
ptr = packed_b + panel_idx * (KC * NR)
```

Performance effect:

- no B gather/scatter in inner loop,
- B load becomes dense vector load,
- K loop streams through a compact panel.

## 7.3 Runtime phase: `MatMul::run`

Inputs:

```text
position_index
position_interval
batch_size = M_run
cpu_num
thread_id
```

The current implementation ignores `position_index` and `position_interval` for this operator and uses `batch_size` as active M.

### 7.3.1 M padding

The kernel assumes fixed `MR`.

Current important assumption:

```text
MR = 3
NR = 32
```

The runtime pads M:

```text
m_pad = ceil(M_run / MR) * MR
```

The implementation expects allocation capacity to cover `m_pad`.

This avoids scalar/tail handling inside the microkernel.

Compiler tradeoff:

| Benefit | Cost |
|---|---|
| Inner loops are branch-free for M tail | Extra compute on padded rows |
| Microkernel fixed shape | Requires capacity discipline |
| Simpler vector kernel | Padding rows must be safe and initialized |

### 7.3.2 Shape constraints

Debug assertions include:

```text
m_pad <= m_max
MB % MR == 0
N % NR == 0
K % KC == 0
```

This means the current backend is tuned for padded/static shapes, not arbitrary dynamic shapes.

### 7.3.3 Tile space

Tile dimensions:

```text
tiles_m = ceil(m_pad / MB)
tiles_n = ceil(N / NB)
tiles_total = tiles_m * tiles_n
```

Each thread receives a contiguous range of tile ids:

```text
[tb, te) = assign(tiles_total, cpu_num, thread_id)
```

Tile id decoding:

```text
tm = tile / tiles_n
tn = tile % tiles_n

m0 = tm * MB
n0 = tn * NB
```

### 7.3.4 Inner loop nesting

The core loop order is:

```text
for tile in assigned_tiles:
  for k0 in 0..K step KC:
    for nt in 0..N_block step NR:
      b_panel = packed_panel_ptr(n0 + nt, k0)

      for mi in 0..M_block step MR:
        a_tile = A[(m0 + mi), k0]
        c_tile = C[(m0 + mi), n0 + nt]

        compute(a_tile, b_panel, c_tile)
```

The most important loop properties:

- thread partitioning is over M/N tiles,
- K is reduced inside each thread,
- B panel is reused across all `mi` inside the tile,
- C tile is updated in place across K panels.

### 7.3.5 Microkernel call

For `f16`, `compute(...)` calls:

```text
kernel::x86_64::f16_512::matmul_block::matmul_block(...)
```

when compiled with:

```text
target_arch = "x86_64"
target_feature = "avx512fp16"
```

Otherwise it falls back to generic matmul.

# 8. AVX-512 FP16 microkernel

File: `src/kernel/x86_64/f16_512/matmul_block.rs`

## 8.1 Kernel shape

The kernel assumes:

```text
A tile: 3 x KC
B panel: KC x 32
C tile: 3 x 32
```

So:

```text
MR = 3
NR = 32
```

For `KC=64`:

| Object | Elements | Bytes at f16 |
|---|---:|---:|
| A tile | `3 * 64 = 192` | 384 B |
| B panel | `64 * 32 = 2048` | 4096 B |
| C tile | `3 * 32 = 96` | 192 B |

This is intentionally L1/L2 friendly.

## 8.2 Kernel algorithm

For each K:

```text
bvec = load 32 FP16 values from B panel
a0   = broadcast A row 0 scalar
a1   = broadcast A row 1 scalar
a2   = broadcast A row 2 scalar

c0 = fma(a0, bvec, c0)
c1 = fma(a1, bvec, c1)
c2 = fma(a2, bvec, c2)
```

At the end:

```text
store c0, c1, c2
```

### 8.2.1 Vector width

AVX-512 FP16 vector:

```text
__m512h = 32 lanes of f16
```

So `NR=32` maps exactly to one vector.

### 8.2.2 FMA count

For one microkernel call:

```text
FMAs = MR * NR * KC = 3 * 32 * KC
```

For `KC = 64`:

```text
FMAs = 3 * 32 * 64 = 6144 fused multiply-adds
FLOP equivalent ~= 12288
```

### 8.2.3 Accumulation precision

The kernel uses `_mm512_fmadd_ph`, so accumulation appears to be in FP16 vector registers, not FP32 accumulation.

That is good for raw throughput but may affect numerical accuracy.

Compiler-performance note:

If target CPUs support AMX-FP16/BF16, an AMX backend could potentially outperform the current AVX-512 FP16 microkernel for larger tiles. The README mentions AMX-capable CPUs, but the inspected code path is primarily AVX-512 FP16, not AMX.

# 9. Fused Q/K/V projection: `MatMul3<T>`

File: `src/compiler/mul/matmul3.rs`

## 9.1 Purpose

`MatMul3` computes:

```text
Q = A * Wq
K = A * Wk
V = A * Wv
```

with weights stored as:

```text
Wq_nt = [Nq  x K]
Wk_nt = [Nkv x K]
Wv_nt = [Nkv x K]
```

Outputs:

```text
Cq = [M x Nq]
Ck = [M x Nkv]
Cv = [M x Nkv]
```

This fuses Q/K/V scheduling into one operator.

## 9.2 Constructor behavior

`MatMul3::new(...)`:

1. stores A/Q/K/V pointers,
2. stores RoPE pointer,
3. stores dimensions,
4. stores tile params,
5. pre-packs Q, K, and V weight panels separately.

Packed buffers:

```text
packed_q
packed_k
packed_v
```

Each uses the same packed B panel format:

```text
[panels_k][panels_n][KC * NR]
```

## 9.3 Runtime task model

`MatMul3::run(...)` computes task counts:

```text
tiles_m = ceil(m_pad / MB)

v_tiles = tiles_m * ceil(Nkv / NB)
k_tiles = v_tiles
q_tiles = tiles_m * ceil(Nq / NB)

total_tasks = v_tiles + k_tiles + q_tiles
```

Task id is decoded as:

```text
0 .. v_tiles                  -> V path
v_tiles .. v_tiles+k_tiles    -> K path
remaining                     -> Q path
```

Each thread receives a contiguous task range via `assign(total_tasks, cpu_num, thread_id)`.

### 9.3.1 Path order

The decode order is:

```text
V first
K second
Q third
```

Potential reason:

- V path has no finalization.
- K/Q path may require RMSNorm/RoPE finalization.
- Splitting by path makes per-path output pointers and packed weights simple.

## 9.4 Per-path tiled GEMM

`gemm_one_path_tiles(...)` performs:

```text
for each assigned tile:
  for k0 in 0..K step KC:
    for nt in 0..N_block step NR:
      for mi in 0..M_block step MR:
        compute1(A tile, B panel, C tile)

        if finalize and last K panel and end of head:
            compute2(C head, rope head)
```

## 9.5 Finalization for Q/K

For K and Q paths:

```text
finalize = true
```

For V path:

```text
finalize = false
```

Finalization triggers when:

```text
k0 + kc_cur == K
offset_in_head + NR == head_dim
```

Meaning:

- only after the final K reduction,
- only at the end of a full head segment.

For `f16`, `compute2(...)` calls:

```text
matmul_finalize_rmsnorm_rope_inplace_3x128(...)
```

This suggests Q/K projection is fused with:

1. matmul accumulation,
2. RMSNorm,
3. RoPE / complex multiplication,
4. in-place storage.

Performance benefit:

- avoids separate operator launch/dispatch for normalization and RoPE,
- keeps head-sized data hot in cache/registers,
- avoids extra intermediate memory writes and reads.

Compiler note:

This is a useful fusion boundary. Q/K post-processing naturally attaches to projection output and can be scheduled at head completion.

# 10. GEMM + local top-k: `MatMulTopK<T>`

File: `src/compiler/mul/matmul_topk.rs`

## 10.1 Purpose

`MatMulTopK` computes matrix products but does not materialize the full `C[M x N]`.

Instead, it keeps top-k results per batch row and per thread.

This is useful for:

- MoE routing,
- logits top-k,
- candidate pruning,
- any operator where only top-k matters.

## 10.2 Data layout

Inputs:

```text
A    = [M x K]
B_nt = [N x K]
```

Outputs:

```text
indices buffer = [batch_max][thread_max][TOPK]
values buffer  = [batch_max][thread_max][TOPK]
```

## 10.3 Internal storage

Constructor allocates:

1. packed B panels,
2. per-thread C tile pool:
   ```text
   [thread_max][MR * NR]
   ```
3. per `(batch, thread)` fixed min-heap:
   ```text
   [batch_max][thread_max]
   ```

The heap writes directly into output value/index buffers.

## 10.4 Runtime flow

For each thread:

1. clear local heaps for true batch rows,
2. partition M/N tile space via `assign(...)`,
3. for each tile:
   - clear thread-local `MR x NR` C tile,
   - loop over K panels and accumulate microkernel result,
   - push each tile result into local heap,
4. sort each local heap descending.

Pseudo-flow:

```text
for batch row:
  clear heap[batch][thread]

for assigned M/N tile:
  for mi in M tile step MR:
    for nt in N tile step NR:
      clear c_tile[MR x NR]

      for k0 in K step KC:
        compute(A tile, B panel, c_tile)

      for r in 0..MR:
        if real batch row:
          for c in 0..NR:
            heap[batch][thread].push(value, col_idx)

for batch row:
  heap[batch][thread].sort_desc()
```

Final top-k across threads is expected to be merged by a later operator or test logic.

## 10.5 Performance benefit

Traditional path:

```text
C = A * B
topk(C)
```

Requires:

- full C materialization,
- full C memory write,
- full C read for top-k,
- large memory bandwidth.

eLLM path:

```text
tile = A_tile * B_panel
heap.push(tile values)
discard tile
```

This reduces memory traffic significantly when `TOPK << N`.

Compiler-performance note:

This is a classic producer-consumer fusion:

```text
GEMM producer + TopK consumer
```

but implemented manually at operator level.

# 11. MoE path

The repo includes multiple MoE-specific operators and kernels:

- `ExpertsSoftmaxNorm`
- `ExpertsMatMulSilu`
- `ExpertsMatMulDown`
- `ExpertsMergeAdd`
- `TopKSoftmax`

## 11.1 Sparse MoE block

File: `src/qwen3_moe/sparse_moe_block.rs`

A sparse MoE block owns:

```text
gate_weight
experts_gate_weight
experts_up_weight
experts_down_weight
```

with shapes roughly:

```text
gate_weight          = [num_experts, hidden_size]
experts_gate_weight  = [num_experts, moe_intermediate_size, hidden_size]
experts_up_weight    = [num_experts, moe_intermediate_size, hidden_size]
experts_down_weight  = [num_experts, hidden_size, moe_intermediate_size]
```

## 11.2 Expert routing buffers

`Tensor::experts_softmax_norm(...)` allocates:

```text
experts_indicator = [num_experts] bool
indice_ptr        = [num_experts, sequence_chunk_size * batch_size] bool
weight_ptr        = [num_experts, sequence_chunk_size * batch_size] T
topk_indices_ptr  = [num_experts_per_tok, sequence_chunk_size * batch_size] usize
```

Performance goal:

- store sparse routing in dense simple buffers,
- avoid dynamic maps/lists in hot kernels,
- make expert activation check pointer-based.

## 11.3 Fused expert gate/up path

`Tensor::experts_matmul_silu_mul_matmul(...)` emits `Operator::ExpertsMatMulSilu`.

Conceptual operation:

```text
gate = A * W_gate
up   = A * W_up
out  = silu(gate) * up
```

The AVX-512 kernel `moe_silu_update_3x32` fuses gate/up accumulation:

```text
for kk in 0..KC:
  bg = load gate_panel[kk, 0:32]
  bu = load up_panel[kk, 0:32]

  a0 = broadcast A row 0 scalar
  a1 = broadcast A row 1 scalar
  a2 = broadcast A row 2 scalar

  gate_acc rows += a * bg
  up_acc rows   += a * bu
```

Benefit:

- one A tile load reused for both gate and up projections,
- better locality,
- fewer passes over activations.

## 11.4 Expert down path

`Tensor::experts_matmul_mul(...)` emits `Operator::ExpertsMatMulDown`.

Conceptually:

```text
down = expert_output * W_down
scaled by routing weight
scatter/accumulate into token output slot
```

The code stores down weights in NT layout:

```text
[E, hidden, intermediate]
```

so per output hidden row has contiguous intermediate dimension.

## 11.5 Expert merge

`Tensor::experts_merge_add(...)` emits `Operator::ExpertsMergeAdd`.

Conceptually:

```text
output = residual + sum(selected expert outputs)
```

This is another fusion point: residual addition is merged with expert output reduction.

# 12. Attention / KV bottleneck strategy

The README states the intended design:

1. static-shape KV cache,
2. non-paged layout,
3. direct coordinate access,
4. contiguous reads along sequence dimension,
5. head-by-head attention.

The performance reasoning is strong:

| Paged KV cache issue | Static contiguous KV benefit |
|---|---|
| pointer chasing | direct offset arithmetic |
| block metadata | no block indirection |
| TLB misses | contiguous spans |
| cache misses | linear prefetchable access |
| fragmented sequence reads | sequence dimension contiguous |
| poor hardware prefetch | predictable stream |

However, in the inspected code, attention implementation appears incomplete / partially commented.

For example, `src/compiler/mul/attention.rs` has placeholder/commented f32 paths. The README also says attention is not fully included in current stage.

So, the KV/attention bottleneck removal is a major architectural claim, but less fully realized in source than the matmul/MoE/operator-queue pieces.

# 13. Mapping from bottlenecks to concrete code mechanisms

## 13.1 Scheduling overhead

### Problem

General serving frameworks often do:

```text
for every token:
  schedule request
  update batch
  route token
  merge/split requests
  launch many small kernels/operators
```

### eLLM mechanism

- Static operator queue.
- Worker threads run same queue.
- Tile partition is computed from thread id.
- No dynamic task queue in hot loop.

Relevant files:

- `src/ptensor/tensor.rs`
- `src/compiler/operator.rs`
- `src/serving/start.rs`
- `src/compiler/assign.rs`

## 13.2 Allocation overhead

### Problem

Decode repeatedly creates:

- Q/K/V tensors,
- attention intermediates,
- MLP activations,
- residual buffers,
- routing buffers.

### eLLM mechanism

- `Tensor::from_cache(...)`
- `Cache::get(...)`
- 64-byte aligned raw allocations
- reuse intermediates across layers by normalized key names

Relevant files:

- `src/memory/cache.rs`
- `src/memory/allocator.rs`
- `src/ptensor/tensor.rs`

## 13.3 Memory bandwidth pressure

### Problem

CPU inference often becomes memory-bandwidth-bound, especially for small-batch matvec/matmul.

### eLLM mechanisms

1. Packed B panels.
2. NT weight layout.
3. Fixed tile shapes.
4. Fused operations.
5. Avoid full materialization in top-k.
6. Contiguous KV design.

Relevant files:

- `src/compiler/mul/matmul.rs`
- `src/compiler/mul/matmul3.rs`
- `src/compiler/mul/matmul_topk.rs`
- `src/kernel/x86_64/f16_512/*`

## 13.4 SIMD underutilization

### Problem

Skinny GEMMs and small batch decode do not naturally fill GPU-style massive parallelism. CPU SIMD also needs careful tiling.

### eLLM mechanism

- Use `MR=3`, `NR=32`.
- `NR=32` exactly matches AVX-512 FP16 lanes.
- B panel is packed as `KC x 32`.
- A scalars are broadcast per row.
- Three C rows are accumulated simultaneously.

Relevant file:

- `src/kernel/x86_64/f16_512/matmul_block.rs`

## 13.5 Top-k materialization

### Problem

Computing full logits then top-k wastes memory bandwidth if only top-k is needed.

### eLLM mechanism

`MatMulTopK`:

- computes `MR x NR` tile,
- pushes tile entries into per-thread heap,
- discards tile,
- stores only local top-k.

Relevant file:

- `src/compiler/mul/matmul_topk.rs`

## 13.6 Q/K/V projection overhead

### Problem

Separate Q, K, V projections require multiple operator dispatches and scheduling phases.

### eLLM mechanism

`MatMul3`:

- packs Q/K/V weights,
- schedules all paths in one operator,
- finalizes Q/K with RMSNorm/RoPE at head boundary.

Relevant file:

- `src/compiler/mul/matmul3.rs`

# 14. Important tuning parameters

The central tiling structure is `MatMulParams`:

```text
a_row_step_macro  = MB
b_row_step_macro  = NB
column_step_macro = KC
a_row_step_micro  = MR
b_row_step_micro  = NR
```

Common values seen in tests:

```text
MR = 3
NR = 32
KC = 64
MB = 3, 6, 24, 96
NB = 32, 64, 128
```

## 14.1 MR

Current fixed microkernel row count:

```text
MR = 3
```

Pros:

- small M tile fits decode/small-batch cases,
- simple register allocation,
- low latency.

Cons:

- unusual MR may underuse available register file for larger M,
- padding overhead for batch sizes not divisible by 3,
- not ideal for all matrix shapes.

## 14.2 NR

Current vector tile width:

```text
NR = 32
```

This exactly maps to:

```text
32 lanes of FP16 in AVX-512
```

Good choice for AVX-512 FP16.

## 14.3 KC

Common:

```text
KC = 64
```

For the main microkernel:

```text
A tile = 3 x 64 x 2 bytes = 384 B
B tile = 64 x 32 x 2 bytes = 4096 B
C tile = 3 x 32 x 2 bytes = 192 B
```

Total active panel footprint is small enough for L1/L2 reuse.

## 14.4 NB

Common:

```text
NB = 32, 64, 128
```

Larger NB gives more N tiles per macro tile and more B-panel reuse. But too large NB can increase cache pressure and reduce scheduling granularity.

## 14.5 MB

Common:

```text
MB = 3, 6, 24, 96
```

Must be divisible by `MR`.

Larger MB improves B-panel reuse across more rows. Smaller MB reduces latency and padding overhead.

# 15. Compiler-performance observations

## 15.1 Good design choices

### Static queue is the right direction

For LLM decode/prefill, operator sequence is mostly fixed. Building it once and replaying it avoids a large amount of framework overhead.

### Weight layout is explicit

The code has moved to a clear contract:

```text
weights are NT: [N x K]
```

This avoids ambiguous transpose costs and helps packed-panel construction.

### Packing B once per operator

Packing in `new()` avoids repeated packing during `run()`.

For fixed model weights, this is appropriate.

### Per-thread scratch buffers

`MatMulTopK` uses per-thread C tile pools. This avoids allocation and false sharing of scratch tiles.

### Local heaps for top-k

For top-k workloads, keeping per-thread heaps avoids materializing full output and avoids synchronization inside the tile loop.

### Core affinity

Pinning threads is useful for cache stability and performance reproducibility.

### Custom barrier is simple and low-latency

The spin barrier is reasonable for short synchronized phases on dedicated inference cores.

## 15.2 Performance risks / limitations

### Barrier after every operator

The runtime waits after each operator. If the operator queue contains many small operators, barrier cost can dominate.

Possible improvements:

- fuse adjacent elementwise operators,
- group independent operators,
- use phase-level barriers rather than per-operator barriers,
- introduce dependency-aware scheduling.

### No NUMA awareness

Core affinity is used, but there is no visible NUMA-aware allocation or first-touch placement strategy.

On multi-socket Xeon/EPYC, this can be a major bottleneck.

Possible improvements:

- pin worker groups per NUMA node,
- partition model weights by NUMA node,
- first-touch initialize packed panels on the target NUMA node,
- avoid cross-socket traffic for hot KV/cache tensors.

### Compile-time target-feature dispatch

The code uses compile-time `#[cfg(target_feature = "avx512fp16")]`.

That means binary performance depends heavily on build flags.

Possible improvements:

- runtime CPU feature dispatch,
- separate kernels for AVX2, AVX-512 BF16, AVX-512 FP16, AMX,
- dispatch table at operator construction.

### README mentions AMX, but code path is AVX-512 FP16

The hardware requirement mentions AMX-capable CPUs, but inspected kernels primarily use AVX-512 FP16 intrinsics.

Possible improvements:

- AMX-FP16/BF16 tile kernels for large GEMMs,
- AVX-512 fallback for smaller/tail shapes,
- shape-based kernel selection.

### FP16 accumulation

The microkernel uses `_mm512_fmadd_ph`, which appears to accumulate in FP16.

This is fast but less accurate than FP32 accumulation.

Possible improvements:

- offer FP32 accumulation kernels for accuracy-sensitive paths,
- mixed strategy: FP16 accumulation for intermediate MoE, FP32 for logits or normalization-sensitive paths.

### Static padding requirements

The matmul paths require:

```text
N % NR == 0
K % KC == 0
MB % MR == 0
M capacity >= padded M
```

This keeps inner loops simple but makes shape handling rigid.

Possible improvements:

- separate tail kernels,
- masked AVX-512 tail kernels,
- offline shape padding in model conversion.

### Work partitioning may imbalance sparse MoE

Static contiguous tile assignment is cheap, but sparse expert activation can create imbalance.

Possible improvements:

- expert-aware tile scheduling,
- per-expert work queues built once per routing result,
- dynamic scheduling only for sparse irregular operators.

### Attention implementation appears incomplete

The project’s strongest architectural story is long-context KV/attention locality, but the current source appears less complete here than for GEMM/MoE.

# 16. Operation-level dataflow summary

## 16.1 Decode iteration

```text
for each decode position:
  for each operator in static queue:
    all threads:
      run operator partition for this thread
    barrier
```

## 16.2 Dense matmul

```text
A[M,K]
B_nt[N,K]
  -> pack B into [K panels][N panels][KC,NR]
  -> assign M/N tiles to threads
  -> each tile:
       for K panel:
         AVX512 3x32 microkernel
  -> C[M,N]
```

## 16.3 Q/K/V projection

```text
A[M,K]
Wq_nt[Nq,K]
Wk_nt[Nkv,K]
Wv_nt[Nkv,K]
  -> pack each weight
  -> flatten V/K/Q tiles into one task space
  -> assign tasks to threads
  -> compute V tiles
  -> compute K tiles + RMS/RoPE finalization
  -> compute Q tiles + RMS/RoPE finalization
```

## 16.4 Top-k matmul

```text
A[M,K]
B_nt[N,K]
  -> pack B
  -> each thread owns C_tile scratch
  -> each thread owns heap[batch][thread]
  -> compute tile
  -> push tile values to heap
  -> discard tile
  -> sorted local top-k
  -> later merge across threads
```

## 16.5 MoE path

```text
hidden states
  -> gate projection
  -> expert top-k + softmax/norm
  -> expert gate/up fused projection
  -> SiLU(gate) * up
  -> expert down projection weighted by route probability
  -> merge selected expert outputs + residual
```

# 17. Why this can beat a generic CPU baseline

A generic CPU inference framework may be slower not because each math kernel is bad, but because decode is dominated by control-path overhead:

```text
scheduling
allocation
temporary tensors
runtime queues
framework indirection
paged KV metadata
small operator dispatches
cache-unfriendly layouts
```

eLLM removes or reduces these by:

```text
static graph
raw pointers
preallocated buffers
fixed layouts
packed panels
core-pinned workers
SIMD kernels
operator fusion
top-k fusion
contiguous KV design
```

This explains the README’s short-context CPU baseline improvement claim: even without beating GPU dense math, reducing overhead can reduce TPOT against a general CPU baseline.

# 18. Why this can help long-context prefill

Long-context prefill is dominated by:

1. parameter movement,
2. KV movement,
3. attention memory access,
4. chunking/synchronization overhead.

GPU issue:

```text
VRAM too small
  -> chunk prompt
  -> repeatedly load/reuse parameters and KV chunks
  -> extra scheduling and synchronization
```

CPU advantage:

```text
large DRAM
large LLC
fewer chunk boundaries
contiguous KV
head-local attention
```

eLLM’s intended long-context path:

```text
reserve large sequence dimension
preallocate static KV
prefill in fewer chunks or one pass
compute attention head-by-head
keep one head's KV hot in cache
avoid page/block metadata
```

This is where the project’s “CPU faster than GPU” claim is most plausible: prefill-heavy, long-context, low/moderate concurrency, memory-capacity-constrained GPU baselines.

# 19. Suggested next measurements for a compiler performance engineer

## 19.1 Microkernel counters

Measure:

- cycles per `3x32xKC` tile,
- FP16 FMA throughput,
- L1/L2/LLC miss rate,
- load/store bandwidth,
- port pressure,
- front-end stalls,
- frequency throttling under AVX-512.

Useful cases:

```text
MR=3, NR=32, KC=64
M = 3, 6, 12, 24, 96
N = 32, 128, 2048
K = 64, 256, 2048
```

## 19.2 Barrier overhead

Measure:

```text
barrier latency vs thread count
barrier percentage of runtime per operator
operator duration distribution
slowest-thread wait time
```

If barrier wait time is high, fuse more operators or coarsen scheduling phases.

## 19.3 Packing amortization

Measure:

```text
time to pack B
number of decode/prefill iterations using packed B
packing memory footprint
NUMA locality of packed B
```

Packing is good only if amortized across many runs.

## 19.4 NUMA locality

On multi-socket systems:

```text
numactl --hardware
perf c2c
uncore memory bandwidth counters
remote vs local DRAM access
```

Check whether packed weights and KV tensors reside on the socket where worker threads run.

## 19.5 Tail/padding overhead

Measure overhead when:

```text
batch_size % 3 != 0
N % 32 != 0
K % 64 != 0
```

Current code assumes/pads many shapes.

## 19.6 Top-k fusion effectiveness

Compare:

```text
full C materialization + top-k
vs
MatMulTopK heap fusion
```

Measure:

- memory bytes written,
- LLC misses,
- cycles per row,
- heap overhead as `TOPK` changes.

## 19.7 Precision tradeoff

Compare:

```text
FP16 accumulate
FP32 accumulate
AMX BF16/FP16 if implemented
```

Track:

- numerical error,
- throughput,
- impact on model quality.

# 20. Potential compiler/backend improvements

## 20.1 Add runtime CPU feature dispatch

Current compile-time feature gating is limiting.

Add:

```text
enum KernelBackend {
  Generic,
  AVX2,
  AVX512F,
  AVX512FP16,
  AMXBF16,
  AMXFP16,
}
```

Dispatch at operator construction.

## 20.2 Add AMX kernels

For Sapphire Rapids / Granite Rapids class CPUs, AMX may provide better dense matmul throughput.

Potential strategy:

- AVX-512 for tiny/tail shapes,
- AMX for larger MB/NB/KC shapes,
- shape-based kernel selection.

## 20.3 Formal memory planner

Current regex/name-based cache reuse is useful but fragile.

A compiler-style memory planner could:

- compute tensor liveness,
- assign buffers by interference graph,
- support in-place ops safely,
- reduce peak memory,
- improve NUMA placement.

## 20.4 Operator fusion pass

Current fusion is manual.

Potential automatic fusions:

```text
matmul + bias/add
matmul + RMSNorm
RMSNorm + RoPE
gate + up + SiLU + mul
expert down + weighted add
residual add + norm
```

## 20.5 Replace global barriers with dependency scheduling

Instead of:

```text
barrier after every operator
```

Use:

```text
barrier only where cross-thread dependency exists
```

or group operators into supernodes.

## 20.6 Tail kernels

Add AVX-512 masked tail support for:

```text
M tail
N tail
K tail
```

This reduces padding overhead and broadens shape support.

## 20.7 Better sparse MoE scheduling

For sparse experts:

- group tokens by expert,
- compact active token lists,
- schedule only active expert tiles,
- use per-expert load balancing,
- avoid scanning inactive expert/token pairs.

# 21. conclude

removes CPU bottlenecks through a combination of:

1. static graph-style execution,
2. preallocated raw-pointer tensor storage,
3. deterministic thread tile partitioning,
4. core-pinned worker threads,
5. cache-line aligned memory,
6. packed NT weight layout,
7. AVX-512 FP16 fixed-shape microkernels,
8. fused Q/K/V projection,
9. fused GEMM+TopK,
10. MoE-specific fused kernels,
11. planned contiguous static KV cache.

The strongest implemented pieces are:

- static operator queue,
- memory cache,
- packed GEMM,
- AVX-512 FP16 matmul,
- `MatMul3`,
- `MatMulTopK`,
- MoE kernels.

The less complete or risky areas are:

- attention/KV implementation maturity,
- lack of NUMA strategy,
- barrier granularity,
- no AMX backend despite AMX hardware target,
- compile-time rather than runtime CPU dispatch,
- shape rigidity around `MR=3`, `NR=32`, `KC` divisibility,
- apparent FP16 accumulation accuracy tradeoff.
