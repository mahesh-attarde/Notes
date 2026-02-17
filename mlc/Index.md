## topic list for CPU inference

### 1) Fundamentals of inference (model-agnostic)
- What “inference” means vs training; latency vs throughput vs cost
- Batch size tradeoffs; online vs offline/batch inference
- Determinism, numerical stability, reproducibility
- Model types and their compute patterns: CNNs, Transformers, RNNs, GNNs, classical ML
- FLOPs vs memory bandwidth: why CPUs are often memory-bound
- Arithmetic intensity, data movement, cache locality basics

### 2) CPU architecture essentials (what impacts inference speed)
- Core count, frequency, turbo behavior, thermal/power limits
- SIMD/vector instruction sets: SSE, AVX, AVX2, AVX-512, AMX (Intel), NEON/SVE (ARM)
- Fused multiply-add (FMA) and mixed-precision execution
- Cache hierarchy (L1/L2/L3), cache lines, prefetching
- Memory subsystem: DDR bandwidth, channels, NUMA, page sizes, huge pages
- SMT/Hyper-Threading effects on inference workloads
- Microarchitecture differences (Intel vs AMD vs ARM server CPUs)

### 3) Parallelism on CPU
- Data parallelism (batching across cores)
- Operator-level parallelism (splitting GEMM/conv across threads)
- Pipeline parallelism and asynchronous execution
- Threading runtimes: OpenMP, TBB, pthreads
- Thread pinning/affinity, core isolation, CPU sets, scheduler behavior (Linux)
- Avoiding oversubscription (app threads vs intra-op threads)

### 4) Key math kernels used in inference
- GEMM/MatMul (SGEMM, int8 GEMM), batched GEMM
- Convolution lowering (im2col), direct conv, Winograd, FFT conv (less common now)
- Attention kernels (QKV projections, softmax, KV cache operations)
- LayerNorm/RMSNorm, softmax, GELU/SiLU, elementwise ops
- Embedding lookups (often memory-bound)
- Sparse vs dense compute; structured sparsity vs unstructured

### 5) Model optimization techniques for CPU
- Quantization:
  - Post-training quantization (PTQ) vs quantization-aware training (QAT)
  - int8 per-tensor vs per-channel; symmetric vs asymmetric
  - activation quantization; dynamic vs static quantization
  - int4/int2 topics (where feasible) and accuracy impacts
- Pruning and sparsity:
  - Magnitude pruning; structured pruning; sparsity-friendly architectures
  - When sparsity helps on CPU (often needs kernel support)
- Distillation and model compression
- Operator fusion (e.g., bias+activation, attention fusion)
- Graph optimizations: constant folding, dead node elimination, layout transforms
- Mixed precision on CPU: BF16/FP16 support (hardware dependent), accumulation types

### 6) Data formats, layouts, and memory behavior
- Tensor layouts: NCHW vs NHWC; blocked layouts (e.g., nChw16c)
- Weight packing, prepacking, cache-friendly blocking
- Alignment, padding, contiguous vs strided access
- Memory allocation strategies; arena allocators; avoiding fragmentation
- KV-cache layout and paging for LLM inference
- Impact of sequence length, context window, and batching on cache/memory

### 7) CPU inference runtimes & compilers (ecosystem)
- ONNX Runtime (CPU EP), execution providers, graph optimizations
- Intel oneDNN (DNNL), MKL, MKL-DNN; AMD BLIS/ZenDNN; ARM Compute Library
- OpenVINO (esp. Intel CPUs)
- PyTorch CPU inference: TorchScript, torch.compile (Inductor), channels-last, quantization
- TensorFlow CPU (XLA, oneDNN integration)
- TVM, IREE, MLIR-based compilation
- Apache TVM / Glow concepts: ahead-of-time vs JIT compilation
- Vendor acceleration: Intel AMX, ARM SVE, Apple Accelerate/Metal (CPU side)

### 8) Model serving on CPU (production concerns)
- Serving frameworks: Triton (CPU backend), TorchServe, TF Serving, BentoML, Ray Serve, FastAPI patterns
- Concurrency models: multi-process vs multi-thread; async request handling
- Batching strategies: static batching, dynamic batching, micro-batching
- Latency SLOs: p50/p95/p99; tail latency causes
- Warmup, caching, and model loading costs
- Multi-model hosting, model multiplexing, routing
- Request/response serialization costs (JSON vs protobuf), zero-copy where possible

### 9) LLM-specific CPU inference topics
- Prefill vs decode phases; why decode is harder to parallelize
- KV cache memory growth, quantized KV cache, cache eviction/paging
- Speculative decoding (CPU-friendly considerations)
- Sampling methods (greedy, top-k, top-p) and CPU overhead
- Long-context handling: sliding window attention, chunked attention
- CPU-optimized LLM engines: llama.cpp concepts, GGUF/quantized formats (topic area)
- Prompt batching and continuous batching for token streaming

### 10) Performance measurement, profiling, and benchmarking
- Benchmark design: representative inputs, warmup, steady-state
- Metrics: tokens/sec, ms/token, latency breakdown, throughput, CPU utilization
- Profilers: perf, VTune, py-spy, flamegraphs, ONNX Runtime profiling, PyTorch profiler
- Kernel-level timing vs end-to-end timing; Amdahl’s law
- Identifying bottlenecks: compute-bound vs memory-bound vs overhead-bound
- Regressions: version pinning, reproducible builds, CI perf tests

### 11) Systems-level tuning on Linux (common for CPU servers)
- NUMA awareness: interleaving vs binding; memory locality
- Huge pages (THP), page faults, TLB misses
- Power/perf governors, turbo settings, c-states, p-states
- Docker/Kubernetes CPU limits, cpuset, throttling effects
- I/O bottlenecks: model download, disk mmap, network overhead
- Using multiple processes (one per socket) vs single process

### 12) Accuracy/quality considerations when optimizing
- Quantization error analysis; calibration datasets
- Outlier channels/activations and mitigation (per-channel, smooth quant, clipping)
- Regression testing: golden outputs, tolerance setting
- Task-level metrics: BLEU, ROUGE, accuracy, F1, perplexity, etc.
- Safety checks for LLMs when changing decoding/sampling implementations

### 13) Security, reliability, and compliance (production inference)
- Model integrity, supply chain, artifact signing
- Sandboxing, resource limits, denial-of-service considerations
- Deterministic inference for auditability
- Data privacy: logging hygiene, PII handling

### 14) Hardware selection and capacity planning
- Sizing: cores, memory capacity, bandwidth, sockets, NUMA
- When CPU is the right choice vs GPU/TPU (cost, availability, latency, small models)
- Cost modeling: $/req, $/token, energy efficiency
- Scaling strategies: vertical vs horizontal; load balancing

### 15) Practical implementation topics (hands-on skills)
- Exporting models: ONNX export, TorchScript export; opset compatibility
- Building an inference pipeline: preprocessing, postprocessing, vectorization
- Custom operators and kernel extensions (C++), SIMD intrinsics basics
- Validating correctness under quantization and fused kernels
- Creating reproducible perf experiments and dashboards


