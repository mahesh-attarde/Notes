# CPU-GPU COmpiler wrt Memory Hierarchy:

**CPU compilers optimize for minimizing latency of a few powerful threads; 
GPU compilers optimize for maximizing throughput of many lightweight threads.**  

# 1. CPU vs GPU: the fundamental compiler difference

## CPU hardware philosophy

A CPU core is designed to make a **single thread fast**.

It has:

- Large caches
- Sophisticated branch prediction
- Out-of-order execution
- Speculative execution
- Large register files
- Deep memory hierarchy
- Hardware prefetchers
- SIMD/vector units
- Coherence between cores

So the CPU compiler asks:

> “How do I expose enough locality, independence, and vector structure so this powerful core can run one or a few threads very fast?”


## GPU hardware philosophy

A GPU is designed to make **many threads collectively fast**.

It has:

- Thousands of lightweight lanes/threads
- SIMT/SIMD execution
- Very high memory bandwidth
- High latency global memory
- Small but fast shared/local memory
- Massive register files
- Hardware scheduling across warps/wavefronts
- Less emphasis on single-thread latency

So the GPU compiler asks:

> “How do I arrange memory, synchronization, and parallel work so many lanes move data efficiently without stalling the whole machine?”


# 2. CPU topic: cache locality

## What it means

CPU caches exploit two ideas:

1. **Temporal locality**  
   If you use data now, you may use it again soon.

2. **Spatial locality**  
   If you use address `x`, you may soon use nearby addresses.

Example:

```cpp
for (int i = 0; i < N; i++)
  sum += A[i];
```

This is good locality because `A[i]`, `A[i+1]`, `A[i+2]` are adjacent.

Bad locality:

```cpp
for (int i = 0; i < N; i++)
  sum += A[random_index[i]];
```

The CPU cannot easily predict or cache this pattern.


## Why CPU compilers care

CPU memory access latency is huge compared with arithmetic.

Approximate idea:

| Operation | Relative cost |
|---|---:|
| Integer add | ~1 cycle |
| L1 cache access | few cycles |
| L2 cache access | ~10 cycles |
| L3 cache access | tens of cycles |
| DRAM access | hundreds of cycles |

So if the compiler can transform a program to reuse cache-resident data, performance improves drastically.


## Compiler techniques

### Loop interchange

Bad for row-major arrays:

```cpp
for (int j = 0; j < M; j++)
  for (int i = 0; i < N; i++)
    sum += A[i][j];
```

Better:

```cpp
for (int i = 0; i < N; i++)
  for (int j = 0; j < M; j++)
    sum += A[i][j];
```

Because C/C++ arrays are row-major, `A[i][j]` with inner `j` accesses adjacent memory.


### Loop fusion

Instead of:

```cpp
for (int i = 0; i < N; i++)
  B[i] = A[i] + 1;

for (int i = 0; i < N; i++)
  C[i] = B[i] * 2;
```

Fuse:

```cpp
for (int i = 0; i < N; i++) {
  B[i] = A[i] + 1;
  C[i] = B[i] * 2;
}
```

This keeps `B[i]` hot in registers/cache.


### Loop tiling/blocking

For matrix multiplication:

```cpp
for (i)
  for (j)
    for (k)
      C[i][j] += A[i][k] * B[k][j];
```

Naively, `B[k][j]` may be repeatedly loaded from memory.

Tiled version:

```cpp
for (ii)
  for (jj)
    for (kk)
      for (i in tile ii)
        for (j in tile jj)
          for (k in tile kk)
            C[i][j] += A[i][k] * B[k][j];
```

The compiler tries to keep small tiles of `A`, `B`, and `C` in cache.


## Why the design is this way

CPU caches are hardware-managed. The compiler does not explicitly move data into L1/L2. Instead, it shapes the access pattern so hardware caches are effective.

So CPU cache-locality optimization is mostly about:

- Changing loop order
- Changing data layout
- Improving reuse distance
- Reducing cache misses
- Making prefetch predictable
- Avoiding false sharing across cores

The compiler’s design is therefore **implicit memory hierarchy optimization**.


# 3. CPU topic: prefetching

## What prefetching does

Prefetching means loading data before it is needed.

Example:

```cpp
for (int i = 0; i < N; i++) {
  prefetch(&A[i + 64]);
  sum += A[i];
}
```

The goal is to overlap memory latency with useful computation.


## Hardware prefetching

Modern CPUs have hardware prefetchers that detect regular patterns:

```cpp
A[i]
A[i+1]
A[i+2]
```

or fixed stride:

```cpp
A[i * 4]
```

The hardware automatically brings future cache lines closer.


## Compiler prefetching

The compiler may insert explicit prefetch instructions when:

- Access pattern is predictable
- Hardware prefetcher may not detect it
- Latency is high enough to justify prefetching
- The prefetched data is likely to be used

Example:

```cpp
for (int i = 0; i < N; i++) {
  __builtin_prefetch(&A[index[i + distance]]);
  sum += A[index[i]];
}
```


## Why CPU compiler design is conservative here

Prefetching can hurt.

If the compiler prefetches too early:

- Data may be evicted before use.

If it prefetches too late:

- It does not hide latency.

If it prefetches useless data:

- It pollutes caches.
- It consumes memory bandwidth.
- It may slow other cores.

So CPU compilers are cautious. Prefetching is profitable only when the compiler can estimate:

- Access distance
- Cache size
- Loop trip count
- Memory latency
- Conflict risk
- Bandwidth pressure


## Why design differs from GPU

On GPUs, latency hiding is often done by **running many warps** while others wait.

On CPUs, there are fewer threads, so latency must be hidden by:

- Caches
- Out-of-order execution
- Speculation
- Prefetching
- Vectorization

So CPU compilers care more about prefetch placement than GPU compilers generally do.


# 4. CPU topic: vectorized memory access

## What it means

Instead of scalar loads:

```cpp
x0 = A[i]
x1 = A[i+1]
x2 = A[i+2]
x3 = A[i+3]
```

Use one vector load:

```cpp
v = load_vector(A + i)
```

Then operate on multiple elements at once.


## CPU SIMD model

CPUs have SIMD units:

- SSE: 128-bit
- AVX: 256-bit
- AVX-512: 512-bit
- NEON/SVE on ARM

A 512-bit vector can hold:

- 16 x 32-bit floats
- 8 x 64-bit doubles
- 64 x 8-bit integers


## Why compiler vectorization is hard on CPUs

The compiler must prove:

1. Memory accesses are independent.
2. Iterations do not depend on each other.
3. Alignment is acceptable.
4. No aliasing prevents reordering.
5. Loop trip count can be handled.
6. Reductions are safe.
7. Floating-point transformations obey language rules.

Example:

```cpp
for (int i = 0; i < N; i++)
  A[i] = B[i] + C[i];
```

This is easy to vectorize if `A`, `B`, and `C` do not overlap.

But:

```cpp
void f(float *A, float *B) {
  for (int i = 0; i < N; i++)
    A[i] = B[i] + 1;
}
```

Could `A` and `B` point to overlapping memory? In C/C++, yes.

So the compiler may need runtime alias checks.


## Why vectorized memory access matters

SIMD arithmetic is useless if memory cannot feed it.

For good vectorization, the compiler prefers:

```cpp
A[i], A[i+1], A[i+2], A[i+3]
```

over:

```cpp
A[index[i]]
```

Contiguous memory gives efficient vector loads/stores.

Irregular memory may require gather/scatter instructions, which are often slower.


## Why design is this way

CPU vector units are powerful but relatively explicit. The compiler must convert scalar loop semantics into vector operations while preserving precise language behavior.

So CPU vectorization is constrained by:

- Language aliasing rules
- Exception behavior
- Floating-point strictness
- Alignment
- Memory dependence
- Control flow
- Target vector width

The compiler design is therefore heavy on **dependence analysis and legality checking**.


# 5. CPU topic: alias analysis

## What alias analysis asks

Can two memory references point to the same location?

Example:

```cpp
*A = *B + 1;
```

Can `A == B`?

If yes, the compiler must be careful.


## Why aliasing matters

Consider:

```cpp
void f(int *a, int *b) {
  *a = 1;
  *b = 2;
  printf("%d", *a);
}
```

If `a` and `b` alias, result is `2`.

If they do not alias, result is `1`.

The compiler cannot freely reorder or cache `*a` unless it knows aliasing behavior.


## Alias analysis enables

### Load/store reordering

```cpp
x = *p;
*q = 10;
y = *p;
```

If `p` and `q` do not alias, compiler can replace `y` with `x`.


### Vectorization

```cpp
for (int i = 0; i < N; i++)
  A[i] = B[i] + C[i];
```

Vectorization is easier if `A`, `B`, and `C` do not overlap.


### Register promotion

```cpp
for (...) {
  x = *p;
  ...
  *p = x + 1;
}
```

If no other pointer can modify `*p`, the compiler may keep it in a register.


## Why CPU compilers invest heavily in alias analysis

CPUs rely on aggressive instruction reordering and memory optimization to make single-thread execution fast.

Alias uncertainty blocks:

- Common subexpression elimination
- Loop-invariant code motion
- Vectorization
- Store forwarding optimization
- Register promotion
- Dead store elimination
- Scheduling

Therefore CPU compilers build sophisticated alias analyses:

- Type-based alias analysis
- Basic alias analysis
- Scoped noalias metadata
- Interprocedural alias analysis
- Escape analysis
- Mod/ref analysis
- MemorySSA-style reasoning


## GPU angle

GPU compilers also need alias analysis, but many GPU programming models expose more structured address spaces:

- Global memory
- Shared memory
- Local/private memory
- Constant memory
- Texture memory

If two pointers are known to be in different address spaces, they cannot alias.

Example:

```cpp
__shared__ float s[];
float *global;
```

A shared-memory pointer and a global-memory pointer are distinct address spaces.

So GPU alias analysis can sometimes benefit from address-space separation.


# 6. CPU topic: loop tiling

## What loop tiling does

Loop tiling breaks large iteration spaces into smaller blocks.

Example: matrix multiplication.

Naive:

```cpp
for (i = 0; i < N; i++)
  for (j = 0; j < N; j++)
    for (k = 0; k < N; k++)
      C[i][j] += A[i][k] * B[k][j];
```

Tiled:

```cpp
for (ii = 0; ii < N; ii += T)
  for (jj = 0; jj < N; jj += T)
    for (kk = 0; kk < N; kk += T)
      for (i = ii; i < ii + T; i++)
        for (j = jj; j < jj + T; j++)
          for (k = kk; k < kk + T; k++)
            C[i][j] += A[i][k] * B[k][j];
```


## Why tiling is essential

Without tiling, working sets may exceed cache size. Data gets loaded repeatedly from DRAM.

With tiling, a block of data is reused many times while still resident in cache.


## Compiler design difficulty

The compiler must choose tile sizes based on:

- L1/L2/L3 cache size
- Cache associativity
- Register pressure
- Vector width
- Memory layout
- Loop bounds
- Parallelism
- Target architecture
- Cost model

This is hard because the optimal tile size differs across machines.


## Why design is this way

Tiling is a compiler attempt to convert slow global-memory reuse into fast cache/register reuse.

For CPUs:

> “Make the hardware-managed cache see a small, reusable working set.”

For GPUs:

> “Explicitly stage data into shared memory and coordinate threads to reuse it.”

That distinction is crucial.

CPU tiling usually targets caches.  
GPU tiling usually targets shared memory/LDS/registers.


# 7. CPU topic: NUMA awareness

## What NUMA means

NUMA = Non-Uniform Memory Access.

On multi-socket systems, memory is physically attached to different CPU sockets.

Example:

```text
Socket 0 ---- local DRAM 0
Socket 1 ---- local DRAM 1
```

A core on socket 0 can access socket 1’s memory, but it is slower.


## Why compilers care

In large servers, memory placement matters.

If a thread running on socket 0 mostly accesses memory allocated on socket 1, performance suffers.


## What compilers can do

Traditional compilers have limited direct NUMA control, but they can help via:

- Loop partitioning
- Data placement hints
- OpenMP scheduling
- First-touch initialization patterns
- Parallel loop transformations
- Avoiding false sharing
- Preserving locality between computation and allocation

Example:

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++)
  A[i] = 0;
```

If each thread initializes the memory it later uses, pages may be placed near that thread’s NUMA node.


## Why this is not purely a compiler problem

NUMA behavior depends on:

- OS page allocation
- Thread scheduling
- Runtime system
- Memory allocator
- Hardware topology
- Application workload

So NUMA optimization is often split between:

- Compiler
- Runtime
- OS
- Programmer annotations
- Libraries


# 8. GPU topic: coalesced global memory access

## What coalescing means

On a GPU, threads execute in groups:

- NVIDIA: warp, often 32 threads
- AMD: wavefront/wave, often 32 or 64 lanes

If adjacent lanes access adjacent memory addresses, hardware can combine them into fewer memory transactions.

Good:

```text
lane 0 -> A[0]
lane 1 -> A[1]
lane 2 -> A[2]
...
lane 31 -> A[31]
```

Bad:

```text
lane 0 -> A[random0]
lane 1 -> A[random1]
lane 2 -> A[random2]
...
lane 31 -> A[random31]
```


## Why GPU compilers care heavily

GPU global memory has:

- Very high bandwidth
- High latency
- Transaction-based access
- Efficiency dependent on access pattern

If a warp accesses contiguous addresses, one or few memory transactions may serve all lanes.

If accesses are scattered, the warp may require many transactions.


## Design reason

A GPU does not optimize one lane in isolation. It optimizes the memory behavior of the entire warp/wave.

So the compiler cares about:

- Thread-to-data mapping
- Access stride
- Vectorization
- Alignment
- Memory layout
- Divergence
- Address calculation

The central GPU memory question is:

> “Do lanes in the same warp touch memory in a pattern the memory system can combine?”

This is different from CPU cache locality.

CPU question:

> “Will this core reuse cache lines soon?”

GPU question:

> “Will these lanes collectively issue efficient memory transactions now?”


# 9. GPU topic: shared-memory bank conflicts

## What shared memory is

Shared memory / LDS is a small, fast, explicitly managed memory region shared by threads in a block/workgroup.

NVIDIA calls it shared memory.  
AMD often calls it LDS.

It is much faster than global memory.


## What banks are

Shared memory is divided into banks.

Example conceptual model:

```text
bank 0
bank 1
bank 2
...
bank 31
```

If all lanes access different banks, accesses proceed efficiently.

If multiple lanes access the same bank, accesses may serialize.


## Example

Good:

```text
lane 0 -> bank 0
lane 1 -> bank 1
lane 2 -> bank 2
...
lane 31 -> bank 31
```

Bad:

```text
lane 0 -> bank 0
lane 1 -> bank 0
lane 2 -> bank 0
...
lane 31 -> bank 0
```

This creates a bank conflict.


## Why compilers care

Shared memory is often used for tiling.

For example, matrix multiplication:

1. Load tile from global memory into shared memory.
2. Synchronize.
3. Reuse tile many times.
4. Store result.

But if shared memory layout causes bank conflicts, the “fast memory” becomes slower.


## Compiler design issue

The compiler may transform layout by adding padding:

```cpp
__shared__ float tile[32][32];
```

May become logically:

```cpp
__shared__ float tile[32][33];
```

The extra column changes bank mapping and avoids conflict.


## Why design is this way

GPU shared memory is explicitly managed because GPUs expose more of the memory hierarchy to software.

CPU caches hide bank/layout details from most programs.  
GPU shared memory makes those details visible because explicit control can deliver huge performance.

So GPU compilers often reason about:

- Bank width
- Bank count
- Thread layout
- Access stride
- Padding
- Swizzling
- Vectorized shared-memory access


# 10. GPU topic: address-space lowering

## What address spaces are

GPU IRs often distinguish memory spaces:

- Private/thread-local memory
- Workgroup/shared memory
- Global memory
- Constant memory
- Generic memory
- Texture/surface memory

Example conceptual IR:

```llvm
ptr addrspace(1)  ; global
ptr addrspace(3)  ; shared/workgroup
ptr addrspace(5)  ; private/local
```


## Why address spaces exist

Different GPU memory spaces have different:

- Latency
- Visibility
- Lifetime
- Synchronization rules
- Instructions
- Caching behavior
- Address widths
- Aliasing properties

A load from global memory may use a different machine instruction than a load from shared memory.


## What lowering means

High-level languages may have abstract memory concepts.

The compiler must lower them to target-specific address spaces.

Example:

```cpp
__shared__ float s[256];
float x = global[i];
```

Compiler must know:

- `s` lives in shared memory.
- `global` lives in global memory.
- temporary scalars live in registers/private memory.


## Why design is this way

Unlike CPUs, where most pointers live in one flat virtual address space, GPUs often have multiple physically and semantically different memory spaces.

So GPU compilers need explicit address-space information to:

- Select correct instructions
- Prove non-aliasing
- Insert correct barriers
- Apply correct memory semantics
- Optimize placement
- Manage lifetimes
- Legalize generic pointers


## Important distinction

CPU:

```text
load ptr
```

Mostly one generic memory model.

GPU:

```text
load_global
load_shared
load_constant
load_private
```

Different memory spaces may imply different machine instructions and rules.


# 11. GPU topic: memory scope and memory semantics

## What memory scope means

Memory scope defines *who* must observe a memory operation.

Possible scopes:

- Single thread
- Warp/subgroup
- Workgroup/block
- Device/GPU
- System/CPU + GPU

Example:

```cpp
atomicAdd(..., scope = device)
```

means the atomic is visible consistently across the GPU device.


## What memory semantics means

Memory semantics define ordering guarantees:

- Relaxed
- Acquire
- Release
- Acquire-release
- Sequentially consistent

Example:

```cpp
store_release(flag, 1);
```

means prior writes become visible before the flag is observed by an acquire load.


## Why GPUs need explicit scope

A GPU has enormous parallelism.

Making every atomic or fence system-wide would be expensive.

So GPU programming models allow narrower synchronization.

Example:

- If threads only communicate within one block, use workgroup scope.
- If communication crosses blocks, use device scope.
- If CPU and GPU communicate, use system scope.


## Compiler design consequence

The compiler must preserve memory ordering while still optimizing.

It must not move loads/stores across barriers or fences incorrectly.

Example:

```cpp
shared[x] = value;
barrier();
y = shared[z];
```

The compiler cannot reorder the load before the barrier.


## Why design is this way

Synchronization cost grows with scope.

A workgroup-scope barrier is cheaper than a device-wide synchronization.  
A device-scope atomic is cheaper than a system-scope atomic.

Therefore GPU compiler IR and lowering need precise scope information.

The design principle is:

> “Make synchronization as narrow as correctness allows.”


# 12. GPU topic: synchronization barriers

## What barriers do

A barrier makes threads wait until all relevant threads reach that point.

Example CUDA-style:

```cpp
shared[tid] = global[i];
__syncthreads();
x = shared[other_tid];
```

The barrier ensures all stores to `shared` are complete before reads happen.


## Why barriers are essential on GPUs

Threads in a block cooperate.

Common pattern:

1. Each thread loads one element.
2. Barrier.
3. Each thread consumes many elements loaded by others.

Without the barrier, one thread could read shared memory before another thread writes it.


## Why barriers are dangerous

Barriers must be reached uniformly.

Bad:

```cpp
if (tid < 16) {
  __syncthreads();
}
```

If only some threads reach the barrier, deadlock may occur.


## Compiler responsibilities

The compiler must understand barriers because they affect:

- Instruction scheduling
- Memory reordering
- Register allocation
- Control-flow legality
- Divergence analysis
- Shared-memory lifetime
- Optimization boundaries


## Why design differs from CPU

CPU threads are usually heavier and synchronized with locks, atomics, condition variables, etc.

GPU threads in a block are lightweight and often explicitly synchronized at fine granularity.

So GPU compiler design treats barriers as central program structure.


# 13. GPU topic: allocation promotion to shared/local memory

## What allocation promotion means

The compiler may move data from slower or less optimal memory to faster memory.

Examples:

- Global memory → shared memory
- Local memory → registers
- Recomputed values → registers
- Reused tiles → shared memory


## GPU “local memory” warning

In GPU terminology, “local memory” can be confusing.

On NVIDIA, “local memory” often means per-thread memory spilled to global memory, not fast local SRAM.

So:

- **Private/register**: fast
- **Shared/LDS**: fast, per block/workgroup
- **Local memory spill**: often slow, backed by global memory


## Why promotion matters

Example:

```cpp
for (...) {
  use A[i] many times;
}
```

If `A[i]` comes from global memory repeatedly, expensive.

Better:

```cpp
tmp = A[i];  // register
for (...) {
  use tmp;
}
```

For cooperative reuse:

```cpp
shared[tid] = global[i];
barrier();
use shared[...];
```


## Compiler difficulty

Promotion requires proving:

- Data is reused enough
- Shared-memory capacity is sufficient
- Register pressure does not explode
- Occupancy remains acceptable
- Synchronization is correct
- No illegal aliasing occurs
- Lifetime fits the target memory space


## Why design is this way

GPU performance often depends on manually or automatically staging data.

But resources are limited:

- Registers per SM/CU
- Shared memory per block
- Maximum resident warps
- Occupancy
- Bank conflicts

Promotion can improve locality but reduce occupancy.

So GPU compiler cost models must balance:

```text
more reuse/locality
vs
less parallel occupancy
```

This is one of the most important GPU compiler tradeoffs.


# 14. GPU topic: vectorized loads/stores

## What vectorized memory means on GPU

A thread may load multiple adjacent elements at once:

```cpp
float4 v = reinterpret_cast<float4*>(A)[i];
```

This can become a wider memory transaction per lane.


## Why useful

Vectorized loads/stores can:

- Reduce instruction count
- Improve memory transaction efficiency
- Improve alignment
- Increase bandwidth utilization
- Match hardware load/store width


## But it is not always good

Vectorized memory can increase:

- Register pressure
- Alignment requirements
- Memory waste
- Occupancy loss
- Spill risk

Example:

```cpp
float4 v;
```

uses four scalar registers conceptually.

If many such vectors exist, register pressure increases.


## CPU vs GPU distinction

CPU vectorization often means:

> “One thread uses SIMD lanes to process multiple data elements.”

GPU vectorization often means:

> “Each lane/thread issues wider memory operations while the warp as a whole remains SIMT.”

Both can use vector instructions, but the execution model differs.


# 15. GPU topic: LDS/shared-memory usage

## What LDS/shared memory is used for

Common uses:

- Matrix multiplication tiles
- Convolution windows
- Reductions
- Prefix sums/scans
- Stencil computations
- Data transposition
- Avoiding repeated global loads
- Cooperative data exchange between threads


## Example: matrix multiplication

Each thread block computes a tile of `C`.

Instead of every thread repeatedly loading from global memory:

1. Load tile of `A` into shared memory.
2. Load tile of `B` into shared memory.
3. Barrier.
4. Compute partial products.
5. Barrier.
6. Load next tile.

This greatly reduces global memory traffic.


## Why compilers care

Shared memory is fast but limited.

Using too much shared memory per block reduces how many blocks can run concurrently.

Example:

```text
SM has 64 KB shared memory.
Kernel uses 16 KB per block.
=> up to 4 blocks resident.

Kernel uses 32 KB per block.
=> up to 2 blocks resident.
```

Less residency can mean less latency hiding.


## Main tradeoff

Shared memory improves reuse, but may reduce occupancy.

The compiler must balance:

- Shared memory usage
- Register usage
- Occupancy
- Bank conflicts
- Barrier cost
- Reuse benefit
- Global memory traffic reduction


# 16. The major CPU/GPU compiler contrast

## CPU compiler mostly optimizes around caches

CPU memory hierarchy is mostly implicit.

The compiler transforms code so the hardware cache system performs well.

Key question:

> “Can I improve locality and vectorization without violating language semantics?”


## GPU compiler often optimizes explicit memory movement

GPU memory hierarchy is more exposed.

The compiler or programmer often decides what goes into:

- Registers
- Shared memory/LDS
- Global memory
- Constant memory

Key question:

> “Can I map many threads to memory so global accesses are coalesced, shared accesses avoid conflicts, and synchronization is correct?”


# 17. Same concept, different meaning

| Concept | CPU compiler meaning | GPU compiler meaning |
|---|---|---|
| Locality | Cache reuse by one/few threads | Coalescing + shared-memory reuse by many lanes |
| Vectorization | SIMD inside a core | SIMT lanes + sometimes per-thread vector ops |
| Tiling | Fit working set in cache | Stage tiles into shared memory |
| Prefetch | Hide latency for few threads | Often replaced by massive thread-level latency hiding |
| Alias analysis | Enable scalar/vector memory reordering | Also helped by explicit address spaces |
| Synchronization | Threads, atomics, locks | Barriers, memory scopes, warp/block/device semantics |
| Register pressure | Affects spills and instruction scheduling | Also affects occupancy heavily |
| Memory hierarchy | Mostly hardware-managed | More explicitly software/compiler-managed |


# 18. Why GPU compilers care less about branch prediction

CPUs have sophisticated branch predictors because one thread’s control flow matters a lot.

GPUs execute warps/waves together.

If lanes diverge:

```cpp
if (thread_id % 2 == 0)
  A();
else
  B();
```

The warp may execute both paths with masks.

So GPU compiler concern is not “predict the branch well” as much as:

- Minimize divergence
- Reconverge efficiently
- Avoid divergent barriers
- Preserve lane masks
- Use predication when profitable

CPU:

> “Will the branch predictor guess right?”

GPU:

> “Will lanes in the same warp take different paths?”


# 19. Why CPU compilers care deeply about instruction latency

CPU has few heavyweight cores. If a dependency chain stalls, performance suffers.

Example:

```cpp
x = f(x);
x = f(x);
x = f(x);
```

This serial dependency limits instruction-level parallelism.

CPU compilers therefore care about:

- Instruction scheduling
- Unrolling
- Software pipelining
- Register allocation
- Reducing dependency chains

GPUs also care, but latency can often be hidden by switching to another ready warp.

So GPU compiler asks more often:

> “Do I have enough occupancy to hide latency?”


# 20. Occupancy: a GPU-specific design pressure

Occupancy means how many warps/waves can be resident on a compute unit.

Occupancy is limited by:

- Registers per thread
- Shared memory per block
- Threads per block
- Hardware maximum warps
- Barriers/resources

If a kernel uses too many registers, fewer warps fit.

This hurts latency hiding.

So GPU compiler register allocation is not only about avoiding spills. It is also about preserving occupancy.

CPU register allocation question:

> “Can I keep values in registers and avoid spills?”

GPU register allocation question:

> “Can I keep values in registers without reducing occupancy too much?”

That difference is huge.


# 21. Design summary

## CPU compiler design is shaped by:

- Fast single-thread performance
- Hardware-managed caches
- Branch prediction
- Out-of-order execution
- SIMD vector units
- Complex aliasing rules
- Cache locality
- Instruction-level parallelism
- NUMA on large systems

CPU compiler philosophy:

> Transform code so a small number of powerful cores see predictable, local, vectorizable work.


## GPU compiler design is shaped by:

- Massive thread parallelism
- SIMT execution
- High global-memory latency
- Very high bandwidth if coalesced
- Explicit shared memory
- Synchronization barriers
- Memory scopes
- Address spaces
- Occupancy
- Divergence
- Bank conflicts

GPU compiler philosophy:

> Map many lightweight threads onto hardware so memory transactions are coalesced, shared memory is reused efficiently, synchronization is correct, and enough warps remain active to hide latency.


# 22. model

For CPU optimization, think:

```text
one/few threads
deep cache hierarchy
make data reuse predictable
make loops vectorizable
reduce branch/memory stalls
```

For GPU optimization, think:

```text
many threads
warp-level behavior
coalesced global memory
shared-memory tiling
avoid divergence
avoid bank conflicts
preserve occupancy
```


# 23. LLVM/MLIR code
| Area | LLVM/MLIR concept |
|---|---|
| CPU loop locality | Loop passes, LoopInterchange, LoopUnroll, LoopVectorize, Polly |
| CPU vectorization | LoopVectorize, SLPVectorizer, VPlan |
| Alias analysis | BasicAA, TBAA, ScopedNoAliasAA, MemorySSA |
| CPU target lowering | X86, AArch64, RISCV backends |
| GPU address spaces | LLVM addrspace, NVPTX, AMDGPU |
| GPU memory lowering | NVPTX/AMDGPU instruction selection |
| GPU shared memory | addrspace lowering, LDS handling, GPU dialect in MLIR |
| GPU barriers | LLVM intrinsics, NVVM/ROCDL, GPU dialect ops |
| GPU tiling | MLIR Linalg, Affine, SCF, GPU, Transform dialect |
| GPU memory coalescing | MLIR vectorization, layout transforms, target-specific lowering |

