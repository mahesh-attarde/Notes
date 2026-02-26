# CPU Inference Fundamentals 
Context
- *inference* (serving/predicting) on CPUs, 
- independent of any single framework (PyTorch/TF/ONNX Runtime)
- emphasizes performance concepts that generalize across models.


## 1) Fundamentals of inference

### What happens during inference (high level)
+ process of applying a trained model’s parameters (weights) to new input data to produce outputs.

A simplified pipeline:

1. **Input acquisition**
   - Receive a request (text/image/tabular/etc.)
2. **Pre-processing**
   - Tokenization, normalization, resizing, feature engineering
3. **Model execution**
   - Matrix multiplies / convolutions / embeddings / activation functions
4. **Post-processing**
   - Softmax, decoding, thresholding, NMS, ranking, formatting
5. **Return response**
   - Possibly with logging/telemetry

### Model-agnostic view: inference as a graph
ML models can be seen as a directed acyclic graph (DAG) of operations:
- **Compute-heavy ops**: GEMM (matrix multiply), convolution
- **Memory-heavy ops**: embedding lookups, layernorm, attention KV-cache reads
- **Control-flow / irregular ops**: beam search, top-k, conditional branches

On CPU, **the same model can be fast or slow** depending on:
- data layout (NCHW vs NHWC, row-major vs col-major)
- vectorization (SIMD: AVX2/AVX-512)
- threading (OpenMP/TBB/pthreads)
- memory locality (cache behavior)
- batch sizing and request pattern

## 2) What “inference” means vs training; latency vs throughput vs cost

### Inference vs training
**Training**
- Objective: minimize loss by updating weights
- Needs: forward pass + backward pass + optimizer step
- Typical properties:
  - much more compute than inference
  - stores activations for backprop (high memory)
  - often uses large batch sizes and accelerators

**Inference**
- Objective: generate predictions using *fixed* weights
- Needs: forward pass only
- Typical properties:
  - can be much cheaper per example
  - must satisfy product constraints: *latency*, *availability*, *cost*
  - frequently runs with small batches (even batch=1)

### Latency vs throughput vs cost
**Latency**: time to complete one request.
- Often measured as:
  - **p50**: median latency
  - **p95/p99**: tail latency (crucial for user experience)

**Throughput**: how many requests per second (RPS) or items/sec.
- With batching, throughput can increase dramatically even if single-item latency rises.

**Cost**: $ per 1K requests, or $ per million tokens, etc.
- CPU cost depends on:
  - instance type (core count, frequency, cache, memory bandwidth)
  - utilization (idle time is wasted cost)
  - software efficiency (vectorization, threading, quantization)

### Example: tradeoff intuition
- A chatbot with interactive users cares about **p95 latency**.
- A nightly batch scoring job cares about **throughput/cost**, and can accept higher latency.

## 3) Batch size tradeoffs; online vs offline/batch inference

### Batch size basics
Batch size = number of inputs processed together in one model execution.

#### Why batching helps
Many operations (GEMM, conv) become more efficient with larger matrices.
- Better SIMD utilization
- Better amortization of overheads (framework calls, scheduling)
- More work per memory fetch (sometimes)

#### Why batching hurts (for online services)
Batching often increases **queueing delay**:
- requests wait to form a batch
- leads to higher tail latency

### Online vs offline inference
**Online (real-time) inference**
- User-facing APIs
- Constraints:
  - tight latency SLOs (e.g., p95 < 200ms)
  - unpredictable traffic
  - must handle spikes gracefully

**Offline / batch inference**
- Backfills, analytics, nightly scoring
- Constraints:
  - maximize throughput
  - minimize cost
  - can use huge batches, asynchronous pipelines

### Practical batching patterns
- **Dynamic batching**: accumulate requests for up to `T` milliseconds or until batch size `B` reached.
- **Micro-batching**: small batches (e.g., 2–16) to balance tail latency and efficiency.
- **Sequence-length bucketing** (for NLP): group similar lengths to reduce padding waste.

## 4) Determinism, numerical stability, reproducibility

### Determinism (same input → same output)
Inference ideally should be deterministic, but can become non-deterministic due to:
- multi-thread scheduling (order of floating-point reductions changes)
- different instruction paths (SIMD kernels, fused ops)
- different libraries / compiler flags (MKL vs OpenBLAS vs oneDNN)
- hardware differences (AVX2 vs AVX-512)

**Key concept:** floating-point addition is not associative:
- `(a + b) + c` may differ slightly from `a + (b + c)`

So if threads sum partial results in different orders, output can vary at small tolerances.

### Numerical stability
Some operations are sensitive:
- softmax on large logits
- layernorm variance computation
- attention score scaling
- long reduction chains (summing many values)

**Stable softmax trick**:
- compute `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

### Reproducibility checklist
To reproduce results across machines/runs:
- pin library versions (BLAS, inference runtime)
- pin model weights and preprocessing code
- fix thread counts (e.g., `OMP_NUM_THREADS`)
- disable certain “fast math” compiler options if exact match is needed
- accept tolerance-based comparisons (e.g., `atol`, `rtol`) in tests

## 5) Model types and their compute patterns

### CNNs (Convolutional Neural Networks)
Common in vision, some audio.
- Heavy ops: convolution → often lowered to GEMM or direct conv kernels
- Access patterns:
  - good spatial locality if implemented well
  - can be compute-heavy, but still can become memory-bound depending on kernel

CPU inference often benefits from:
- optimized conv kernels (oneDNN)
- quantization (INT8)
- careful layout (NHWC sometimes better for certain kernels)

### Transformers
Common in NLP, LLMs, vision transformers.
Main components:
- embeddings (memory-heavy)
- attention (mix of compute & memory, plus KV cache reads in decoding)
- MLP blocks (large GEMMs)

**Two phases:**
1. **Prefill (prompt processing)**: parallel across tokens, larger matrices → higher throughput.
2. **Decode (token-by-token)**: batch often small, lots of memory traffic (KV cache) → harder on CPU.

CPU inference pain points:
- KV cache bandwidth and cache misses
- small GEMMs during decode (less efficient)
- softmax + layernorm overhead relative to GEMM

### RNNs (LSTM/GRU)
- Sequential dependency: time steps processed one after another
- Less parallelism than Transformers
- Often latency-bound by sequential chain; batching helps but not always possible online.

### GNNs (Graph Neural Networks)
- Irregular memory access (neighbors vary)
- Sparse operations and scatter/gather patterns
- Often strongly **memory-bound** on CPU.

### Classical ML (linear/logistic regression, trees, boosting)
- Linear models: dot-products → memory bandwidth can dominate
- Tree ensembles (XGBoost/LightGBM):
  - branch-heavy; may suffer from branch misprediction
  - can be fast if feature access is contiguous and trees are optimized
- Often extremely competitive on CPU with minimal latency.

## 6) FLOPs vs memory bandwidth: why CPUs are often memory-bound

### FLOPs (compute capability)
Roughly: how many floating-point operations per second the CPU can execute.
Modern CPUs have wide SIMD and multiple cores → high peak FLOPs.

### Memory bandwidth (data delivery capability)
How quickly the CPU can read/write data from DRAM.
If your computation needs to fetch lots of data per operation, performance is limited by bandwidth.

### The roofline idea (intuitive)
Performance is bounded by the minimum of:
- **Compute roof** (peak FLOPs)
- **Memory roof** (bandwidth × arithmetic intensity)

If arithmetic intensity is low, you hit the memory roof first.

### Why inference often becomes memory-bound
Inference frequently involves:
- reading large weight matrices
- reading activations
- writing intermediate results
If weights don’t fit in cache, you keep streaming from DRAM.

Common culprits:
- large embeddings
- transformer KV cache during decode
- layernorm / elementwise ops over big tensors
- small batch sizes (less reuse of weights per unit time)

## 7) Arithmetic intensity, data movement, cache locality basics

### Arithmetic intensity (AI)
**AI = (number of arithmetic operations) / (bytes moved from memory)**

- High AI → more compute per byte → more likely compute-bound.
- Low AI → less compute per byte → more likely memory-bound.

#### Example intuition: dot product
For a dot product of length `N`:
- operations: ~`2N` FLOPs (multiply + add)
- data: read `N` floats from A + `N` floats from B (and maybe write 1 result)
- AI is modest; often bandwidth-sensitive for large N.

#### Example intuition: matrix multiply (GEMM)
A well-tiled GEMM reuses blocks of A and B many times from cache.
- AI can be high → compute-bound when implemented well.
This is why GEMM is a “good” CPU workload.

### Data movement costs
Accessing memory has a hierarchy:
1. Registers (fastest)
2. L1 cache
3. L2 cache
4. L3 cache
5. DRAM (slowest)

If your working set fits in caches, you can be far faster.

### Cache locality
- **Temporal locality**: reuse the same data soon (good for caches).
- **Spatial locality**: access contiguous memory (cache lines fetch chunks).

Common CPU inference optimizations target locality:
- tiling / blocking (operate on small blocks that fit in cache)
- operator fusion (reduce writing/reading intermediate tensors)
- layout transformations to enable sequential memory access
- quantization to reduce bytes moved (e.g., INT8 weights instead of FP32)

## Perspective

When CPU inference is slow, usually one (or more) is happening:
- **Memory-bound**: spending time waiting on DRAM
- **Poor vectorization**: not using SIMD effectively
- **Threading overhead**: too many threads, contention, NUMA effects
- **Small-batch inefficiency**: tiny GEMMs / overhead dominates
- **Pre/post-processing dominates**: tokenization, JSON parsing, resizing

# Deep dive: Latency vs Throughput (Queueing + Dynamic Batching)

+ *why* batching improves throughput but can worsen latency (especially p95/p99)
+  how **dynamic batching** tries to balance both.

## 1) Definitions you will measure in a serving system
Let a request arrive at time `t_arrive`.
- **Queueing delay (Wq)**: time waiting *before* model execution starts  
  `Wq = t_start - t_arrive`
- **Service time (S)**: time spent doing actual compute (and maybe in-process pre/post)  
  `S = t_finish - t_start`
- **Total latency / response time (R)**:  
  `R = Wq + S = t_finish - t_arrive`
- **Throughput**:
  - requests/sec (RPS), or items/sec
  - for LLMs, often tokens/sec

In practice, track percentiles:
- p50 latency: typical experience
- p95/p99 latency: tail, strongly impacted by queueing and batching

## 2) Why throughput and latency conflict

### A. Micro-batching improves compute efficiency
Many ML kernels (GEMM/conv) run more efficiently with batch size > 1:
- better SIMD utilization
- better cache reuse / blocking
- less overhead per request
So **service time per item** often decreases with batch size.

### B. But batching introduces intentional waiting
To form a batch, you often wait for more requests to arrive.
That adds **queueing delay** even when the CPU is idle-ish, and increases tail latency.

**Key idea:**  
Batching trades *some* latency for *higher* throughput / lower cost per request.

## 3) A simple queueing model (intuitive, not overly mathematical)

Assume:
- Requests arrive at average rate **λ** (requests/sec).
- The server can process requests at average rate **μ** (requests/sec).
- Utilization is `ρ = λ / μ`.

As `ρ → 1` (system near saturation):
- queueing delay can grow very fast
- p95/p99 explode even if average service time is unchanged

### Important practical takeaway
Even without batching:
- if you run CPU inference at ~90–95% utilization continuously, tail latency usually becomes unacceptable.

So production systems often target lower utilization (e.g., 50–70%) if latency SLOs are strict.


## 4) Concrete example: batching speeds up compute but can increase latency

### Setup 
Suppose your model on CPU has these measured service times:

| Batch size B | Batch compute time `T(B)` | Per-item service time `T(B)/B` |
|---:|---:|---:|
| 1  | 10 ms  | 10.0 ms |
| 4  | 22 ms  | 5.5 ms |
| 8  | 36 ms  | 4.5 ms |
| 16 | 64 ms  | 4.0 ms |

This is typical: bigger batches amortize overhead and improve kernel efficiency.

### What throughput would be (compute-only)
Throughput ≈ `B / T(B)`

- B=1  → `1 / 10ms`  ≈ **100 req/s**
- B=4  → `4 / 22ms`  ≈ **182 req/s**
- B=8  → `8 / 36ms`  ≈ **222 req/s**
- B=16 → `16 / 64ms` ≈ **250 req/s**

So batching increases throughput and can reduce cost per request.

### But what happens to latency online?
If traffic is sparse, forming a batch costs waiting time.

If arrivals are ~100 req/s (1 request every 10 ms on average):
- To gather B=8 requests, the *average* time span covering those arrivals is ~70 ms (from first to eighth), and the first request might wait close to that.
- Then you still pay compute time ~36 ms.

So early requests in the batch might see:
- waiting: up to ~70 ms
- compute: 36 ms
- total: ~106 ms (for the first request)
while the last request might wait ~0 ms + 36 ms.

This creates **latency spread** and worsens p95/p99.


## 5) Dynamic batching: what it is

A **dynamic batcher** tries to batch requests only when it’s beneficial, using two knobs:

1. **max_batch_size (Bmax)**: don’t exceed this size  
2. **max_delay (Tmax)**: don’t wait longer than this to form a batch

Pseudo-policy:
- start a batch when the first request arrives
- keep adding requests until:
  - batch size == Bmax, OR
  - waiting time since first request == Tmax
- then execute the batch

This turns batching into a controlled form of queueing.

### Why dynamic batching helps
- At high traffic: batches fill quickly → minimal waiting, high throughput
- At low traffic: you hit `Tmax` quickly → avoid extreme delays


## 7) Dynamic batching example with numbers

Use earlier service times. Set:
- `Bmax = 8`
- `Tmax = 10 ms`

### Case A: High load (λ high)
Say requests arrive every ~1 ms on average.
- Filling 8 requests takes ~7 ms
- The first request waits ~7 ms max (often less), which is acceptable
- Compute is 36 ms
- Total for first request: ~43 ms
- Throughput is high (near the batched throughput)

### Case B: Low load (λ low)
Say requests arrive every ~15 ms on average.
- In 10 ms you likely get only 1 request
- Batcher hits `Tmax`, runs batch size 1
- Total latency ≈ 10 ms compute + small wait
- Throughput is lower, but latency stays bounded

So `Tmax` acts like a “latency fuse”.

## 8) Choosing `Bmax` and `Tmax` (practical guidance)

### Start from your product SLO
If p95 latency must be < 100 ms:
- you can’t set `Tmax=50ms` unless compute is tiny

A simple budget method:
- Let `R_budget` = max allowed latency (e.g., 100 ms)
- Let `S_est` = typical compute time (e.g., 36 ms for B=8)
- Then max queueing budget is roughly:
  - `Wq_budget ≈ R_budget - S_est - network - pre/post`

If network + pre/post is 10 ms:
- `Wq_budget ≈ 100 - 36 - 10 = 54 ms`
So choose `Tmax` comfortably below that (e.g., 10–30 ms), leaving headroom.

### Observe the “knee” in efficiency vs batch size
Plot `T(B)/B` vs `B`.
- Often you get big gains from B=1→4, smaller gains after that.
- Pick Bmax near the knee (e.g., 4 or 8) to avoid hurting latency for diminishing returns.

## 9) CPU-specific notes (why batching behavior can differ from GPU)

On CPU:
- Many models become **memory-bandwidth-bound**.
- Increasing batch size does not always scale throughput linearly, because you saturate memory bandwidth or caches.
- For transformer *decode*, batch size may not help much because the workload is dominated by KV-cache reads and small GEMMs.

So always measure: batching wins are model- and phase-dependent.

## 10) Minimum Experiment
To make this real, measure:
1. **Service time curve** `T(B)` for B ∈ {1,2,4,8,16}
   - same sequence length / same input size
   - record p50 compute time

2. For a simulated online stream:
   - generate arrivals with configurable λ (Poisson is a common approximation)
   - implement dynamic batching (`Bmax`, `Tmax`)
   - measure p50/p95/p99 total latency and throughput

3. Plot:
   - throughput vs p95 latency as you vary `Tmax` and `Bmax`
   - choose an operating point that meets SLO with minimal cost


