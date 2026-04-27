### GPU-CPU Compiler (Day 0)
+ CPUs optimize for low-latency, complex control flow and strong single-thread performance
+ GPUs optimize for massive throughput, SIMD/SIMT execution, memory coalescing, and occupancy.

## 1. Execution model

### CPU
A CPU compiler targets:
- Few hardware threads
- Out-of-order execution
- Deep branch prediction
- Large caches
- Sophisticated scalar pipelines
- Vector units such as SSE, AVX, SVE, NEON
The compiler optimizes a relatively small number of threads for low latency.

Typical CPU model:
```text
process
  └── thread
        └── scalar/vector instructions
```

### GPU
A GPU compiler targets:
- Thousands to millions of logical threads
- SIMT/SIMD-style execution
- Warps/wavefronts/subgroups
- Explicit memory hierarchy
- High memory bandwidth
- High latency hidden by massive parallelism

Typical GPU model:

```text
grid / dispatch
  └── block / workgroup
        └── warp / wavefront / subgroup
              └── lane / thread
```

## 2. Control Flow : Scalar vs SIMT
### CPU
Branches are mostly handled by:
- Branch prediction
- Speculation
- Out-of-order execution
- If-conversion when profitable
- Vector predication on some targets
CPU compiler concern:
```cpp
if (x > 0)
  a = b + c;
else
  a = b - c;
```
The compiler Tasks:
- Should this stay as a branch?
- Should it become a conditional move?
- Should it be vectorized with masks?
### GPU
On GPUs, many lanes execute together. If lanes take different control-flow paths, this causes divergence.
Example:

```cpp
if (thread_id % 2 == 0)
  do_A();
else
  do_B();
```

Within one warp/wavefront, some lanes go to `do_A`, others to `do_B`.

The GPU may execute:

```text
execute A with mask for even lanes
execute B with mask for odd lanes
```

So the compiler must reason about:

- Divergence analysis
- Reconvergence
- Execution masks
- Predicated instructions
- Uniform vs varying values
- Structured vs unstructured control flow

## 3. Uniform vs divergent values

A key GPU compiler concept is whether a value is uniform or divergent.

### Uniform value

Same for every lane in a warp/subgroup.

Example:

```cpp
int x = blockIdx.x;
```

Usually uniform across all threads in a block.

### Divergent value

Different for different lanes.

Example:

```cpp
int x = threadIdx.x;
```

Usually different per thread.

Why this matters:

- Uniform branches are cheap.
- Divergent branches are expensive.
- Uniform loads may use scalar memory paths on some GPUs.
- Divergent addresses affect memory coalescing.
- Uniform values can often be placed in scalar registers.
- Divergent values require vector registers.

On AMDGPU, for example, the compiler has to distinguish SGPRs and VGPRs:

| Register type | Meaning |
|---|---|
| SGPR | Scalar general-purpose register, one value per wave |
| VGPR | Vector general-purpose register, one value per lane |

This affects instruction selection, register allocation, scheduling, and ABI lowering.

## 4. Memory hierarchy
### CPU memory model
Typical CPU hierarchy:

```text
registers
L1 cache
L2 cache
L3 cache
DRAM
```

CPU compilers optimize for:

- Cache locality
- Prefetching
- Vectorized memory access
- Alias analysis
- Loop tiling
- NUMA awareness in some systems

### GPU memory model

Typical GPU hierarchy:

```text
registers
shared/local memory
L1/cache
L2
global memory
constant memory
texture memory
host memory
```

Terminology differs by platform:

| Concept | CUDA | OpenCL/SYCL-ish |
|---|---|---|
| Per-thread registers | registers | private memory |
| Per-block scratchpad | shared memory | local memory |
| Device DRAM | global memory | global memory |
| Read-only broadcast area | constant memory | constant memory |

GPU compilers care heavily about:

- Coalesced global memory access
- Shared-memory bank conflicts
- Address-space lowering
- Memory scope and memory semantics
- Synchronization barriers
- Allocation promotion to shared/local memory
- Vectorized loads/stores
- LDS/shared-memory usage

Example:

```cpp
float x = A[threadIdx.x];
```

Good: adjacent lanes load adjacent addresses.

```cpp
float x = A[threadIdx.x * stride];
```

Potentially bad: non-coalesced loads.

A GPU compiler may try to detect and optimize these patterns, but many optimizations rely on source-level structure or higher-level IR.

## 5. Register pressure and occupancy

CPU compilers care about register pressure because spills are expensive.
GPU compilers care about register pressure because it affects occupancy.

### CPU

Too many live values causes spills:

```text
more live values → stack spills → slower code
```

### GPU

Too many registers per thread reduces how many warps/waves can run concurrently:

```text
more registers per thread
  → fewer resident warps/waves
  → lower latency hiding
  → possibly slower kernel
```

So GPU compilers often trade:

- Instruction count
- Register pressure
- Occupancy
- Memory latency hiding
- Spill cost

A transformation that improves CPU performance may hurt GPU performance if it increases per-thread register usage too much.

Example:

```text
unroll loop heavily
  CPU: maybe good due to reduced branch overhead and more ILP
  GPU: maybe bad due to register pressure and reduced occupancy
```

## 6. Vectorization vs SIMT lowering

### CPU compiler

CPU vectorization is usually an optimization.

The compiler starts from scalar code and tries to create SIMD operations:

```cpp
for (int i = 0; i < n; ++i)
  C[i] = A[i] + B[i];
```

Becomes something like:

```text
load vector A[i:i+7]
load vector B[i:i+7]
vector add
store vector C[i:i+7]
```

Key CPU passes:

- Loop vectorization
- SLP vectorization
- Cost modeling
- Masked vectorization
- Interleaving
- Runtime alias checks
- Remainder loop generation

### GPU compiler

GPU source is often already expressed as many logical scalar threads:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];
```

The compiler does not necessarily vectorize in the same way. Instead, the hardware groups scalar-looking threads into warps/waves.

The compiler asks:

- Which values are uniform/divergent?
- How to map thread IDs?
- How to handle barriers?
- How to lower address spaces?
- How to manage execution masks?
- How to schedule for latency hiding?
- How to allocate SGPR/VGPR/register files?

So CPU SIMD vectorization and GPU SIMT compilation are related but not identical.

## 7. Cost models are very different

A CPU cost model may care about:

- Latency
- Throughput
- Branch prediction
- Cache misses
- Vector width
- Instruction fusion
- Micro-op count
- Port pressure
- Function call overhead

A GPU cost model may care about:

- Occupancy
- Warp divergence
- Memory coalescing
- Shared-memory bank conflicts
- Register usage
- Instruction issue rate
- Tensor/matrix core utilization
- Barrier cost
- Global memory latency
- Wave-level operations
- LDS/shared memory capacity

For example, loop unrolling:

| Transformation | CPU effect | GPU effect |
|---|---|---|
| More unrolling | More ILP, fewer branches | More ILP but higher register pressure |
| Less unrolling | Smaller code, less pressure | More occupancy but maybe less latency hiding |
| Vectorization | Often essential | May be less central in SIMT model |
| Predication | Avoids branches | Can avoid divergence but may waste lanes |

## 8. Address spaces

GPU IRs usually need explicit address spaces.

Example LLVM-style idea:

```llvm
addrspace(1) = global memory
addrspace(3) = shared/local memory
addrspace(4) = constant memory
addrspace(5) = private memory
```

The exact mapping depends on target.

CPU code usually uses a mostly flat address space, while GPU code distinguishes:

- Global memory
- Shared/local memory
- Constant memory
- Private memory
- Generic pointers

This affects:

- Type legalization
- Alias analysis
- Pointer casts
- Load/store selection
- Memory instruction selection
- ABI lowering

Generic pointer lowering is particularly important in CUDA, HIP, OpenCL, and SYCL-style compilers.

## 9. Synchronization and memory semantics

CPU compiler synchronization issues include:

- Atomics
- Locks
- Memory ordering
- Thread sanitizer instrumentation
- Volatile and fences
- C++ memory model

GPU compilers additionally deal with:

- Workgroup barriers
- Subgroup barriers
- Device-wide synchronization limitations
- Memory scopes
- Memory spaces
- Warp-synchronous programming
- Cooperative groups
- Atomic scopes

Example scopes:

```text
thread
warp/subgroup
workgroup/block
device
system
```

A GPU atomic may be:

```text
atomic add, workgroup scope
atomic add, device scope
atomic add, system scope
```

The compiler must lower these to target-specific instructions and fences.

## 10. Function calls and recursion

### CPU

Function calls are cheap enough to be common. CPUs have mature ABIs, stacks, return address prediction, and calling conventions.

### GPU

Historically, GPU kernels preferred:

- Inlining
- No recursion
- Limited function pointers
- Restricted dynamic allocation
- Restricted stack usage

Modern GPUs support more, but calls can still be costly because they affect:

- Register pressure
- Stack/private memory usage
- Occupancy
- Control-flow complexity

Therefore, GPU compilers tend to inline aggressively, especially inside kernels.

## 11. ABI and kernel launching

CPU compilation usually produces:

```text
object file
functions
standard ABI
linked executable or shared library
```

GPU compilation often involves:

```text
host code
device code
kernel metadata
fat binary
runtime registration
device-specific object format
JIT or ahead-of-time compilation
```

CUDA/HIP/SYCL/OpenMP target offloading commonly split compilation into:

```text
host compilation path
device compilation path
bundling
linking
runtime launch
```

A GPU compiler must generate not only instructions but also metadata:

- Kernel argument layout
- Required workgroup size
- Shared memory usage
- Register usage
- Occupancy hints
- Target features
- Code object version
- Calling convention
- Address-space information

## 12. Scheduling

### CPU scheduling

CPU instruction scheduling cares about:

- Pipeline latency
- Issue ports
- Micro-op fusion
- Register renaming
- Out-of-order execution
- Branch prediction
- Cache hierarchy

Because the hardware is sophisticated, the compiler often relies on the CPU to recover some scheduling inefficiencies.

### GPU scheduling

GPU hardware also schedules, but the compiler may need to expose enough independent work while controlling register usage.

GPU scheduling cares about:

- Long memory latency
- Occupancy
- Dual issue
- VALU/SALU balance
- Tensor-core pipelines
- Memory pipeline pressure
- Barrier placement
- Wait states
- Scoreboarding
- Instruction grouping

On AMDGPU, for instance, scheduling between scalar and vector instructions can be important. On NVIDIA, hiding latency across warps is central.

## 13. Backend lowering differences

In LLVM terms, a CPU backend and GPU backend both do instruction selection, register allocation, and scheduling, but GPU targets often need extra target-specific handling.

CPU backend concerns:

- Legalize scalar/vector types
- Select machine instructions
- Register allocation
- Stack frame layout
- Calling convention
- Branch relaxation
- Object emission

GPU backend concerns:

- Kernel calling convention
- Thread/block intrinsic lowering
- Address-space lowering
- Divergence analysis
- Uniform register allocation
- Execution masks
- Barrier handling
- Occupancy calculation
- Code-object metadata
- Special memory instructions
- Texture/sampler/image operations
- Matrix/tensor instructions

