# Workload characterization
describing *what a program does to the machine* in a way that helps you predict performance, pick hardware, and choose optimizations. 

Two of the most useful lenses are:
1) **Arithmetic Intensity (AI)**: how much compute you do per unit of data moved  
2) **Computational / access pattern**: the structure of operations and memory/communication behavior (regular vs irregular, vectorizable vs branchy, etc.)

## 1) Arithmetic Intensity (AI): the core idea
**Definition**  
Arithmetic intensity is typically:

\[
AI = \frac{\text{useful arithmetic operations (FLOPs)}}{\text{bytes transferred}}
\]

- “Bytes transferred” is often measured at a specific level:
  - **DRAM bytes** (main memory traffic) → common for roofline modeling
  - **Cache bytes** (L1/L2/L3 traffic) → useful for tuning kernels
- AI is **not just algorithmic**; it depends on implementation (tiling, reuse, precision, layout).

**Why it matters**  
It helps decide whether your code is:
- **Compute-bound** (limited by peak FLOPs)
- **Memory-bound** (limited by memory bandwidth)

A common mental model is the **Roofline Model**:
- Peak performance ceiling: `Peak FLOPs/s`
- Bandwidth ceiling: `Memory BW (bytes/s) * AI (FLOPs/byte)`
- Attainable performance ≈ `min(peak_flops, bw*AI)`

### Example A: SAXPY (y = a*x + y)
Per element:
- FLOPs: 2 (mul + add)
- Memory (typical, from DRAM): read x (4B) + read y (4B) + write y (4B) = **12 bytes** (float)
- AI ≈ 2 / 12 = **0.167 FLOPs/byte** → very low → **memory-bound** on most CPUs.

### Example B: Dense matrix multiply (GEMM)
For C = A·B (NxN):
- FLOPs ≈ 2N³
- Data ≈ 3N² elements (A,B,C), bytes ≈ 3N²·8 for FP64 (roughly)
- AI grows ~ O(N). With blocking, you reuse A/B in cache heavily → high AI → often **compute-bound**.

## 2) Computational pattern: “how” work is structured
Arithmetic intensity alone doesn’t tell you everything. Two kernels can have the same AI but behave very differently due to the pattern.
### 2.1 Memory access pattern (spatial/temporal locality)
Key ideas:
- **Spatial locality**: contiguous access (good for caches, prefetchers)
- **Temporal locality**: reuse the same data soon (enables caching)
- **Stride**: step between accessed elements; stride-1 is ideal

Examples:
- Row-wise array traversal: great locality
- Column-wise traversal in row-major arrays: poor locality (large stride)
- Sparse gather/scatter: irregular locality (hard to prefetch)

### 2.2 Control-flow pattern
- **Branchy / divergent** code hurts:
  - CPU branch mispredict penalties
  - SIMD lane inefficiency (vectorization less effective)
- “Straight-line” loops vectorize well.

Example:
- Filtering with unpredictable `if` conditions often becomes **branch/mispredict-bound** even if AI is moderate.

### 2.3 Parallelism pattern
- **Data parallel** (same operation over many elements) → SIMD + threads friendly
- **Task parallel** (independent tasks) → scheduling overhead matters
- **Pipeline** patterns → latency and synchronization can dominate

### 2.4 Communication / synchronization pattern
In multithreading:
- Locks/atomics create serialization
- Barriers create phase coupling
- False sharing (two threads update different variables on same cache line) can destroy scaling


## 3) What CPU specs matter (and why)
When characterizing a workload on a CPU, these specs map directly to bottlenecks:
### 3.1 Compute throughput
- **Core count** and **frequency**
- **SIMD width** (AVX2 256-bit, AVX-512 512-bit)
- **FMA units** (fused multiply-add doubles FLOP rate for many kernels)
- **FP32 vs FP64 throughput** differs (FP64 often lower on some CPUs; not always)

**Peak FLOPs/s (rough idea)**  
\[
\text{Peak} \approx \text{cores} \times \text{GHz} \times \text{SIMD lanes} \times \text{ops/cycle}
\]
Where ops/cycle depends on how many vector FMAs can retire per cycle.

Why this matters: high-AI kernels (GEMM, FFT in-cache sections) trend toward this limit.

### 3.2 Memory system
- **Memory bandwidth** (GB/s): sets the ceiling for low-AI kernels (streaming)
- **Memory latency** (ns): critical for pointer-chasing, hash tables, graph workloads
- **Cache sizes** (L1/L2/L3) and associativity
- **NUMA topology** (multi-socket): remote memory access is slower

### 3.3 Execution / pipeline behavior
- **Instruction mix**: integer vs FP, scalar vs vector
- **Out-of-order window** helps hide latency if there’s enough independent work
- **Prefetchers** help for regular patterns; fail for irregular patterns

## 4) Common workload “classes” (AI + pattern together)

### Class 1: Streaming / bandwidth-bound (low AI, regular)
Examples:
- SAXPY, memcpy, simple reductions
Traits:
- Regular stride-1 access
- Few FLOPs per byte
Typical limit:
- DRAM bandwidth (and sometimes store bandwidth)

### Class 2: Cache-friendly compute kernels (moderate-high AI, regular)
Examples:
- Blocked GEMM, stencil computations (when tiled)
Traits:
- Strong temporal locality due to reuse
- Vectorizable inner loops
Limit:
- compute throughput or L1/L2 bandwidth

### Class 3: Latency-bound irregular memory (very low effective AI)
Examples:
- Hash table lookups, graph BFS, pointer chasing in linked lists/trees
Traits:
- Poor spatial locality
- Many cache misses
- Prefetching ineffective
Limit:
- memory latency + limited memory-level parallelism (MLP)

### Class 4: Branch / control dominated (AI not the main story)
Examples:
- Parsing (JSON), interpreters/VMs, complex conditionals
Traits:
- High branch mispredict
- Small working sets sometimes fit in cache but still slow
Limit:
- front-end/branch predictor + pipeline flush cost

### Class 5: Mixed / real applications
Most real workloads are mixtures: e.g., ML inference might have:
- GEMM/conv (compute-bound)
- softmax/normalization (bandwidth-bound)
- embedding table lookups (latency-bound)

## 5) How to *measure* and report workload characterization

A good characterization usually includes:

1) **Operational intensity (FLOPs/byte)** at DRAM and possibly at cache levels  
2) **Working set size** vs cache sizes  
3) **Access pattern**: sequential/stride/gather-scatter, reuse distance  
4) **Parallel scaling**: threads vs speedup, NUMA sensitivity  
5) **Vectorization**: SIMD efficiency, alignment, remainder handling  
6) **Bottleneck metrics** (from profilers):
   - IPC (instructions per cycle)
   - cache miss rates (L1/L2/L3, TLB misses)
   - branch mispredict rate
   - memory bandwidth utilization
   - stalled cycles (backend vs frontend)

Tools people use (CPU side): Linux `perf`, VTune, `likwid`, `pcm`, `perf stat -d`, etc.

## 6) Small “back-of-the-envelope” examples tying it together

### Example: Dot product
- FLOPs: 2N (mul+add)
- Bytes: 8N for FP32 (read two arrays) or 16N for FP64
- AI (FP32): ~2 / 8 = **0.25 FLOPs/byte**
Even with perfect SIMD, it tends to be **bandwidth-bound**.

### Example: 7-point stencil (3D) without/with blocking
Naively:
- Each grid update reads multiple neighbors → lots of traffic
With blocking:
- Neighbors reused from cache → effective bytes per update drops
- AI increases → can move from memory-bound toward compute-bound

## 7) A simple template to describe a workload (practical checklist)

When you describe a workload, you can fill in:

- **Problem size / data type**: N, precision (FP32/FP64), structure (dense/sparse)
- **Operations**: FLOPs per element / per iteration
- **Memory traffic**: bytes per element at DRAM (estimate) + expected reuse
- **AI estimate**: FLOPs/byte
- **Access pattern**: stride-1? blocked? random?
- **Parallelism**: SIMD (yes/no), threads (yes/no), synchronization frequency
- **Likely bound**: compute / bandwidth / latency / branch



## 1) Arithmetic Intensity (AI): the core idea
### Example A: SAXPY (y = a*x + y)
Per element:
- FLOPs: 2 (mul + add)
- Memory (typical, from DRAM): read x (4B) + read y (4B) + write y (4B) = **12 bytes** (float)
- AI ≈ 2 / 12 = **0.167 FLOPs/byte** → very low → **memory-bound** on most CPUs.

### Example B: Dense matrix multiply (GEMM)
For C = A·B (NxN):
- FLOPs ≈ 2N³
- Data ≈ 3N² elements (A,B,C), bytes ≈ 3N²·8 for FP64 (roughly)
- AI grows ~ O(N). With blocking, you reuse A/B in cache heavily → high AI → often **compute-bound**.

## 7) How people optimize SpMV (and what each targets)

### 7.1 Improve locality of `x` accesses
- **Reordering** rows/columns (RCM, METIS/graph partitioning) to reduce bandwidth/working set and increase reuse.
- **Blocking** (BSR format) when sparsity has small dense blocks; reduces index overhead and increases compute per fetched x line.

### 7.2 Reduce metadata and improve SIMD friendliness
- Use **SELL-C-σ**, ELLPACK variants, or sliced formats for better vectorization (trade memory overhead for regularity).
- BSR helps especially for PDE systems and coupled fields.

### 7.3 Increase parallel efficiency / NUMA correctness
- First-touch allocate `A`, `x`, `y` properly per NUMA node.
- Partition by nnz, not just by rows, to reduce imbalance.

### 7.4 Hide latency (limited on CPUs)
- Software prefetching sometimes helps but is tricky because of indirection.
- Ensure enough independent work per thread to increase MLP (unrolling, multiple rows at once), but returns diminish.

