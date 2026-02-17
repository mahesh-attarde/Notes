Learning Notes
# CNN CPU Inference 
## 1. Defs
### Latency (single/low-batch responsiveness)  (CNN)
Latency is the time from input ready to output produced.
Common CNN latency modes:
- **Single image (batch=1)**: most interactive use-cases.
- **Small batch (2–8)**: common compromise for server level workloads.
Key percentiles:
- **p50**: median.
- **p95 / p99**: tail latency (often dominated by scheduling, NUMA, cache misses).
### Throughput (requests/sec or images/sec at higher concurrency/batching) (CNN)
Throughput is usually measured as:
- **images/sec** at a fixed accuracy and input size.
- **requests/sec** at a fixed batch size and concurrency.
Often achieved by:
- larger batch sizes (if allowed),
- multi-threading across cores,
- running multiple processes/instances.

### Core CNN workload characterization om CPU
Most CNN inference is dominated by:
- **Convolutions** (direct or lowered to GEMM),
- **1x1 convolutions** (GEMM-like),
- **Depthwise convolutions** (often memory-bound),
- plus activation, pooling, normalization, elementwise ops.

CPU performance depends on a balance of:
- **vector compute** (SIMD / matrix units),
- **cache locality and memory bandwidth**,
- **kernel library quality** (oneDNN, ZenDNN, ACL),
- **threading/NUMA**.

## 2. CNN Ops mix and its implications
### Conv types and CPU behavior
- **1x1 conv / pointwise**: highly optimized GEMM kernels that benefits heavily from vectorization and cache-aware packing.
- **3x3 / 5x5 conv**: may use direct conv, Winograd (less common in newer toolchains), or implicit GEMM.
- **Depthwise conv**: frequently **memory-bound** and sensitive to layout and cache.
- **Residual blocks**: add/activation fusion matters (reduces memory traffic).

### Layout 
Common layouts:
- **NCHW**: often training-friendly.
- **NHWC**: often faster on some CPU backends due to vectorization choices.
- **Blocked layouts** (e.g., `nChw16c`, `OIhw16i16o`): frequently used internally by oneDNN / ZenDNN to maximize SIMD utilization.

For CNNs on CPU, the runtime frequently:
1. transforms weights to packed blocked format (one-time cost),
2. runs conv kernels in blocked layout.

If you measure latency, **separate**:
- cold start (includes weight packing),
- warmed (steady state).

## 3. CPU Features for CNN inference (latency & throughput)
- **AVX2 / AVX-512**: wide SIMD; great for FP32/FP16/BF16.
- **VNNI** (AVX-512 VNNI / AVX2 VNNI on newer parts): accelerates **dot products**.
- **AMX** (Advanced Matrix Extensions, on newer server/workstation): very strong for  **BF16** matrix/tile ops (huge for conv/GEMM-heavy CNNs).
- Strong ecosystem integration: **oneDNN**, **OpenVINO**, ONNX Runtime Intel optimizations.
### 3.2 CPU latency profile (CNN)
Strengths:
- **Excellent batch=1 latency** when kernels are highly optimized and thread counts are tuned.
- Strong with 8 bit quantized CNNs (PTQ/QAT) due to VNNI/AMX.
Latency pitfalls and fixes:
- **Too many threads** can *hurt* latency (thread wakeups, contention).
  - Tune `intra_op_num_threads` / `OMP_NUM_THREADS`.
- **NUMA**: cross-socket memory access can spike p99.
  - Pin to a socket and allocate memory locally.
- **Frequency scaling / power**: turbo behavior affects latency variance.
  - Use performance governor; avoid aggressive power saving in prod.
Recommended latency tuning (conceptual):
- Use **1 instance per socket** (or per NUMA domain) for stable latency.
- For batch=1, start with:
  - `threads ≈ physical_cores_per_socket / 2` (then tune),
  - pin threads (compact affinity) to improve cache locality.

>TODO: Examples

### 3.3 CPU throughput profile (CNN)
Strengths:
- High throughput scaling with:
  - many cores,
  - good GEMM/conv kernels,
  - batched inference (if allowed),
  - multiple concurrent model instances.
Throughput best practices:
- Increase **concurrency** rather than huge batch if tail latency matters.
- For offline throughput:
  - larger batch + all cores can maximize images/sec.
- Consider **INT8** to reduce memory bandwidth pressure and boost throughput.

### 3.4 CPU quantization (CNN)
- **INT8 PTQ** works well for CNNs with calibration.
- Per-channel weight quantization improves accuracy.
- With AMX/VNNI:
  - lower quantized kernels can be dramatically faster than FP32.
- Watch out for:
  - accuracy regression in depthwise-heavy mobile nets,
  - unsupported ops falling back to slower paths.

## 4.  X CPUs for CNN inference (latency & throughput)
- Strong **core counts** and **memory bandwidth** per system (platform dependent).
Latency pitfalls and fixes:
- over-threading increases latency variance.
- **NUMA topology** can be more complex on high-core-count multi-chip designs:
  - memory locality is critical,
  - Chiplets : cross-die or cross-socket traffic can hurt p99.

Recommended latency tuning:
- Pin to NUMA nodes; use **local allocation**.
- Consider **more instances with fewer threads each** to stabilize p99.

Typical strengths:
- High throughput due to:
  - many cores,
  - strong memory subsystem (platform dependent),
  - scaling with multiple instances.
- CNNs that are compute-heavy (e.g., ResNet-like) can scale well.

Throughput best practices:
- Use instance replication (process-level) to exploit core count without contention.
- For offline throughput, use all cores but ensure:
  - memory bandwidth is not saturated by unnecessary copies,

# CPU Inference for Transformers / LLMs

 LLM inference is different from CNN inference in terms of  **autoregressive decoding**, where each next token depends on the previous one.

## 1. Definitions (LLM context)

### Latency 
LLM latency is multi-dimensional. Always break it down into:

1. **Time to First Token (TTFT)**  
   Time from request start to first generated token output.
   - Dominated by **prefill** (prompt processing) and system overhead.
   - Highly sensitive to prompt length and batching strategy.

2. **Time per Output Token (TPOT)** (often ms/token)  
   Time to generate each subsequent token during decoding.
   - Dominated by **decode** phase compute and KV-cache memory access.
   - Harder to parallelize; often determines “streaming responsiveness”.

3. **End-to-end latency**  
   TTFT + (output_tokens × TPOT) + overhead (tokenization, network, etc.)

Tail latency (p95/p99) is especially important in LLM serving because:
- requests vary widely in prompt length and output length,
- batching introduces queueing delay,
- NUMA and memory pressure can cause spikes.

### Throughput (LLM)
Throughput is typically measured as:
- **tokens/sec** (system-wide) at a target latency SLO,
- or **requests/sec** at fixed prompt/output lengths.

Important: “Higher batch” can increase tokens/sec, but may worsen TTFT and p99 due to queueing and larger working sets.

## 2. LLM compute anatomy on CPU

### 2.1 Prefill vs decode
- **Prefill**:
  - processes the entire prompt (sequence length = prompt length),
  - uses attention over the full prompt,
  - is more parallelizable and can benefit from batching.

- **Decode**:
  - generates tokens one step at a time,
  - attention uses **KV cache**; each new token attends to all previous tokens,
  - far less parallel across time; performance is often **memory-bandwidth / cache** constrained.

### 2.2 Dominant kernels
For most Transformer blocks (decoder-only LLMs), dominant operations are:
- **MatMul/GEMM** for:
  - QKV projections,
  - output projection,
  - MLP up/down projections.
- **Attention**:
  - softmax + weighted sum over KV cache,
  - KV cache read bandwidth becomes a major limiter as context grows.
- **LayerNorm/RMSNorm** and elementwise activations:
  - memory-bound but important at scale due to frequent repetition.

### 2.3 Why CPU LLM inference is hard
- Decode is sequential: limited by per-token work. WHY?
- KV cache grows with: TODO:Examples
  - sequence length,
  - number of layers,
  - hidden size,
  - precision of KV storage.
- CPUs are strong at general compute but comparatively limited by:
  - memory bandwidth vs GPU HBM,
  - smaller on-chip caches relative to KV working set.

## 3. Precision formats and quantization (LLM-specific)

### 3.1 Common weight precisions on CPU
- **FP32**: baseline, usually too slow for large LLMs.
- **BF16**: widely used on CPU where supported; often a sweet spot for accuracy/perf.
- 8-Bit MX: good speedups; accuracy can remain strong with proper methods.
- 4-bit MX: often essential for running larger models on CPU memory budgets; kernel and format support matter.

### 3.2 KV cache precision
- KV cache can be stored as:
  - FP16/BF16 (higher accuracy, more bandwidth),
  - 8-bit (lower bandwidth, potential quality impact),
  - specialized quantized KV (implementation-specific).
KV cache precision directly impacts decode TPOT.

### 3.3 Quantization approaches that matter
- PTQ vs QAT; for LLMs, specialized PTQ methods are common.
- Per-channel weight quantization is generally better.
- Outlier handling is critical for LLM accuracy (some layers have large-magnitude channels).
- Partial quantization (some ops fall back) can destroy expected gains.

## 7. LLM serving strategies on CPU 

### 7.1 Batching: static vs continuous
- **Static batching**: fixed batch size; simple but can inflate tail latency.
- **Continuous/dynamic batching**: merges token-generation steps across requests:
  - raises tokens/sec,
  - can control TTFT vs throughput trade-off via batching windows and max batch size.

Key parameters:
- max batch size (tokens or sequences),
- batching window (queue delay budget),
- maximum context length admitted.

### 7.2 Multi-instance vs single instance
For CPU LLM serving, **multiple smaller instances** often outperform one giant instance because:
- better NUMA locality,
- less lock contention,
- more predictable tail latency,
- easier CPU pinning.

Rule of thumb:
- Prefer **one instance per NUMA node** for stable p95/p99.

### 7.3 Memory management
- KV cache dominates memory footprint for long contexts.
- Avoid frequent allocations:
  - use arenas,
  - preallocate KV buffers,
  - reuse request structures.

### 7.4 Tokenization and runtime overhead
In CPU serving, tokenization can be non-trivial:
- Use fast tokenizers (compiled implementations) and reuse buffers.
- Separate tokenization threads from decode threads if it causes contention.

---


## 8. LLM benchmark matrix (what to test on CPU)
Fix
- Model: e.g., 7B / 13B class (or your target size)
- Quantization: FP32 vs BF16 vs ...
- Context length: 256, 1k, 4k, 8k (and your real)
- Workload mixes:
  - short prompt / long output,
  - long prompt / short output,
  - long prompt / long output (worst-case)
- Metrics:
  - TTFT p50/p95/p99,
  - TPOT p50/p95/p99,
  - tokens/sec aggregate,
  - memory usage (RSS),
  - bandwidth utilization if measurable.

Also record:
- threads per instance,
- number of instances,
- NUMA binding policy,
- batching policy and window.

