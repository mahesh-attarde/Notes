# CPU vs GPU Control Flow in Compilers
## Index
2. [Why Control Flow Is Different on CPUs and GPUs](#2-why-control-flow-is-different-on-cpus-and-gpus)
3. [Architectural Control-Flow Model](#3-architectural-control-flow-model)
4. [Compiler IR View of Control Flow](#4-compiler-ir-view-of-control-flow)
5. [Common LLVM Control-Flow Passes](#5-common-llvm-control-flow-passes)
6. [CPU-Oriented Control-Flow Passes](#6-cpu-oriented-control-flow-passes)
7. [GPU-Oriented Control-Flow Passes](#7-gpu-oriented-control-flow-passes)
8. [Detailed CPU Pass Examples and Transformations](#8-detailed-cpu-pass-examples-and-transformations)
9. [Detailed GPU Pass Examples and Transformations](#9-detailed-gpu-pass-examples-and-transformations)
10. [Control Flow and Architectural Parameters](#10-control-flow-and-architectural-parameters)
11. [CPU vs GPU Transformation Trade-Off Matrix](#11-cpu-vs-gpu-transformation-trade-off-matrix)
12. [Case Study 1: If-Then-Else](#12-case-study-1-if-then-else)
13. [Case Study 2: Switch Lowering](#13-case-study-2-switch-lowering)
14. [Case Study 3: Loop Unswitching](#14-case-study-3-loop-unswitching)
15. [Case Study 4: Loop Unrolling](#15-case-study-4-loop-unrolling)
16. [Case Study 5: Divergent Early Exit](#16-case-study-5-divergent-early-exit)
17. [Case Study 6: Irreducible Control Flow](#17-case-study-6-irreducible-control-flow)
18. [CPU Machine-Level Control Flow](#18-cpu-machine-level-control-flow)
19. [GPU Machine-Level Control Flow](#19-gpu-machine-level-control-flow)
20. [AMDGPU Control-Flow Pipeline Example](#20-amdgpu-control-flow-pipeline-example)
21. [NVPTX Control-Flow Pipeline Notes](#21-nvptx-control-flow-pipeline-notes)
22. [SPIR-V and Structured Control Flow](#22-spir-v-and-structured-control-flow)
23. [MLIR Control-Flow Lowering View](#23-mlir-control-flow-lowering-view)
24. [Performance Modeling Checklist](#24-performance-modeling-checklist)
25. [Practical Debugging Techniques](#25-practical-debugging-techniques)
26. [Recommended Reading Path in LLVM](#26-recommended-reading-path-in-llvm)
27. [Summary](#27-summary)


# Nutshell
Control flow is one of the biggest areas where **CPU and GPU compilers diverge**.
A CPU compiler usually optimizes control flow for: (  Minimize mispredictions and maximize instruction locality / ILP.)
- Branch prediction
- Instruction-cache locality
- Fallthrough layout
- Speculation
- Out-of-order execution
- Loop canonicalization
- Vectorization enablement
- Low-latency execution
A GPU compiler usually optimizes control flow for: (  Minimize divergence and execution-mask overhead while preserving occupancy.)
- Divergence reduction
- Uniform vs divergent branch separation
- Reconvergence
- Execution-mask manipulation
- Structured control flow
- Occupancy
- Register pressure
- Memory coalescing preservation
- Barrier correctness
- SIMD/SIMT lane utilization


# 2. Why Control Flow Is Different on CPUs and GPUs
## CPU Execution Model
A CPU core is optimized for fast execution of one or a few instruction streams.
Typical CPU features:
- Deep pipelines
- Branch predictors
- Speculative execution
- Out-of-order execution
- Register renaming
- Large coherent caches
- Low-latency scalar execution
- SIMD units used explicitly by vector instructions

CPU branch handling:

```text
if branch predicted correctly:
  cost is low

if branch mispredicted:
  pipeline flush
  frontend restart
  high latency penalty
```

Approximate branch cost model:

```text
Expected branch cost =
  P(correct) * cheap_cost +
  P(mispredict) * mispredict_penalty
```

For modern CPUs, mispredict penalties can be significant.

## GPU Execution Model
A GPU executes many lightweight threads grouped into warps, wavefronts, or subgroups.
Typical GPU features:
- SIMT/SIMD execution
- Thousands of resident threads
- High memory latency hidden by occupancy
- Explicit hierarchy: thread, warp, block/workgroup, grid
- Execution masks
- Divergence and reconvergence
- Large register files
- Memory coalescing requirements

GPU branch handling:

```text
if all lanes take same path:
  uniform branch
  cheap

if lanes split across paths:
  divergent branch
  execute each path under different masks
  inactive lanes waste cycles
```

Approximate divergence cost model:

```text
Divergent if cost ≈ cost(then path under mask) + cost(else path under mask) + mask/reconvergence overhead
```

If half the lanes take one path and half take the other, the warp may execute both paths serially.

# 3. Architectural Control-Flow Model
## CPU Control Flow

```text
Instruction Stream
    |
    v
Branch Predictor
    |
    v
Speculative Fetch / Decode
    |
    v
Out-of-Order Core
    |
    v
Retire
```

Important CPU architectural parameters:

| Parameter | Control-flow relevance |
|---|---|
| Branch predictor accuracy | Determines branch penalty |
| Pipeline depth | Deeper pipelines increase mispredict cost |
| I-cache size | Code expansion may hurt locality |
| BTB capacity | Too many branches can hurt prediction |
| µop cache | Helps repeated branchy code |
| Out-of-order window | Can hide some control-flow latency |
| SIMD width | Influences if-conversion/vector predication |
| Frontend bandwidth | Block layout affects fetch/decode efficiency |

## GPU Control Flow

```text
Kernel
  |
  v
Thread Blocks / Workgroups
  |
  v
Warps / Wavefronts / Subgroups
  |
  v
SIMT Lanes
  |
  v
Execution Mask
```

Important GPU architectural parameters:

| Parameter | Control-flow relevance |
|---|---|
| Warp/wave size | Granularity of divergence |
| Execution mask register | Tracks active lanes |
| Reconvergence mechanism | Determines divergent branch handling |
| Register pressure | Affects occupancy |
| Occupancy | Hides latency |
| Shared memory usage | Affects resident blocks |
| Barrier cost | Control-flow-sensitive correctness/performance |
| SIMT stack / token stack | May track divergent paths |
| Scalar/vector register split | Important on AMDGPU |
| Uniform branch support | Allows cheap scalar branches |
| Predication support | Avoids branches but may execute useless work |

# 4. Compiler IR View of Control Flow
Most middle-end compilers represent control flow as a graph of basic blocks.
Example source:

```c
int f(int x) {
  if (x > 0)
    return x + 1;
  else
    return x - 1;
}
```

LLVM-like IR:

```llvm
define i32 @f(i32 %x) {
entry:
  %cond = icmp sgt i32 %x, 0
  br i1 %cond, label %then, label %else

then:
  %a = add i32 %x, 1
  br label %merge

else:
  %b = sub i32 %x, 1
  br label %merge

merge:
  %r = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %r
}
```

Control-flow passes may transform this to:

```llvm
define i32 @f(i32 %x) {
entry:
  %cond = icmp sgt i32 %x, 0
  %a = add i32 %x, 1
  %b = sub i32 %x, 1
  %r = select i1 %cond, i32 %a, i32 %b
  ret i32 %r
}
```

On CPUs, this can avoid a branch.

On GPUs, this can avoid divergent branch control flow but may execute both arithmetic paths.


# 5. Common LLVM Control-Flow Passes

These passes are not strictly CPU-only or GPU-only. Many are used in both contexts.

| Pass | Category | Purpose |
|---|---|---|
| `SimplifyCFGPass` | IR CFG simplification | Merge blocks, fold branches, simplify switches |
| `JumpThreadingPass` | IR branch optimization | Thread known branch conditions through predecessors |
| `DFAJumpThreadingPass` | IR branch optimization | Thread switch/state-machine-like control flow |
| `LowerSwitchPass` | IR lowering | Lower `switch` to branches/jump tables |
| `LoopSimplifyPass` | Loop canonicalization | Add preheaders, simplify loop CFG |
| `LCSSAPass` | Loop SSA canonicalization | Make loop exits use LCSSA PHIs |
| `LoopRotationPass` | Loop canonicalization | Rotate loops into preferred shape |
| `SimpleLoopUnswitchPass` | Loop optimization | Move loop-invariant branches outside loops |
| `LoopUnrollPass` | Loop optimization | Duplicate loop body to reduce branch overhead / expose ILP |
| `LoopUnrollAndJamPass` | Loop optimization | Unroll outer loop and fuse inner bodies |
| `LoopSimplifyCFGPass` | Loop CFG simplification | Simplify branch structure inside loops |
| `LoopDeletionPass` | Dead loop removal | Remove loops without side effects |
| `SCCPPass` | Sparse conditional constant propagation | Remove unreachable paths |
| `CorrelatedValuePropagationPass` | Value/CFG simplification | Fold branches using correlated facts |
| `ADCEPass` | Aggressive DCE | Remove dead blocks and dead control flow |
| `FixIrreduciblePass` | CFG repair | Convert irreducible CFG into reducible form |
| `UnifyLoopExitsPass` | Loop CFG normalization | Canonicalize loop exits |
| `StructurizeCFGPass` | Structured CFG | Important for GPU / SPIR-V-like targets |

# 6. CPU-Oriented Control-Flow Passes

CPU-oriented pipelines heavily use common IR passes plus machine-level block and branch optimizations.

## CPU IR-Level Passes

| Pass | Main CPU goal |
|---|---|
| `SimplifyCFGPass` | Reduce branches, fold blocks, improve canonical form |
| `JumpThreadingPass` | Eliminate redundant dynamic branches |
| `DFAJumpThreadingPass` | Optimize state machines and switch loops |
| `LowerSwitchPass` | Prepare efficient branch trees/jump tables |
| `SimpleLoopUnswitchPass` | Hoist invariant branches from loops |
| `LoopRotationPass` | Improve loop form for vectorization and branch prediction |
| `LoopUnrollPass` | Reduce branch overhead and expose ILP |
| `LoopFlattenPass` | Simplify nested loop control flow |
| `IndVarSimplifyPass` | Simplify loop exits and induction-based conditions |
| `CorrelatedValuePropagationPass` | Fold branches using known facts |

## CPU Machine-Level Passes

| Pass / concept | Main CPU goal |
|---|---|
| `BranchFolderPass` | Remove redundant branches, merge blocks |
| `MachineBlockPlacementPass` | Improve fallthrough layout and I-cache locality |
| `TailDuplicatePass` | Duplicate small blocks to avoid branches |
| `IfConverterPass` | Convert branches to predicated instructions where profitable |
| `EarlyIfConverterPass` | Early machine if-conversion |
| `BranchRelaxationPass` | Expand or relax branches when displacement is too large |
| `UnreachableMachineBlockElimPass` | Remove dead machine blocks |
| `MachineBlockFrequencyInfo` | Guide block placement |
| `MachineBranchProbabilityInfo` | Guide branch layout and prediction-sensitive transforms |

# 7. GPU-Oriented Control-Flow Passes

GPU pipelines use many common IR passes, but also need GPU-specific passes.

## GPU IR-Level Passes

| Pass / analysis | Main GPU goal |
|---|---|
| `DivergenceAnalysis` | Determine values/branches that differ per lane |
| `UniformityAnalysis` | Determine values/branches uniform across lanes |
| `StructurizeCFGPass` | Convert CFG into structured regions |
| `FixIrreduciblePass` | Remove irreducible control flow before structurization |
| `UnifyLoopExitsPass` | Normalize loop exits for structured lowering |
| `AMDGPUUnifyDivergentExitNodesPass` | Merge divergent exits for AMDGPU structurization |
| `AMDGPUAnnotateUniformValuesPass` | Annotate uniform branches/values |
| `SIAnnotateControlFlowPass` | Annotate AMDGPU-specific control-flow constructs |
| `AMDGPURewriteUndefForPHIPass` | Fix PHI behavior after structurization |
| `LowerSwitchPass` | Lower switches before GPU control-flow lowering |
| `FlattenCFGPass` | Simplify CFG before structurization/lowering |

## GPU Machine-Level Passes

| Pass | Main GPU goal |
|---|---|
| `SILowerControlFlow` | Lower AMDGPU structured control flow to machine form |
| `SIOptimizeExecMasking` | Optimize execution mask operations |
| `SIOptimizeExecMaskingPreRA` | Optimize mask operations before register allocation |
| `SILateBranchLowering` | Final AMDGPU branch lowering |
| `SIWholeQuadMode` | Handle whole-quad mode for pixel shaders |
| `SILowerI1Copies` | Lower boolean copies/predicates |
| `SILowerWWMCopies` | Lower whole-wave mode copy operations |
| `R600MachineCFGStructurizer` | R600-specific CFG structurization |
| `R600ControlFlowFinalizer` | Finalize R600 control flow |


# 8. Detailed CPU Pass Examples and Transformations


## 8.1 `SimplifyCFGPass`

### Purpose

`SimplifyCFGPass` simplifies the control-flow graph.

Typical transformations:

- Merge blocks with single predecessor/successor
- Remove unreachable blocks
- Fold branches with constant conditions
- Convert branch diamonds to `select`
- Simplify switches
- Hoist/sink common instructions
- Eliminate empty blocks


## Example: Constant Branch Folding

### Before

```llvm
entry:
  br i1 true, label %then, label %else

then:
  call void @foo()
  br label %exit

else:
  call void @bar()
  br label %exit

exit:
  ret void
```

### After

```llvm
entry:
  br label %then

then:
  call void @foo()
  br label %exit

exit:
  ret void
```

### CPU impact

| Architectural parameter | Impact |
|---|---|
| Branch predictor | Fewer branches |
| I-cache | Less code if dead block removed |
| Pipeline | Fewer control transfers |
| Compile-time canonicalization | Enables more DCE |

### GPU impact

| Architectural parameter | Impact |
|---|---|
| Divergence | Removes impossible divergent path |
| Execution mask | Fewer mask changes |
| Occupancy | Usually unchanged |
| Code size | Reduced |

## Example: Diamond to Select

### Before

```llvm
entry:
  %cond = icmp sgt i32 %x, 0
  br i1 %cond, label %then, label %else

then:
  %a = add i32 %x, 1
  br label %merge

else:
  %b = sub i32 %x, 1
  br label %merge

merge:
  %r = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %r
```

### After

```llvm
entry:
  %cond = icmp sgt i32 %x, 0
  %a = add i32 %x, 1
  %b = sub i32 %x, 1
  %r = select i1 %cond, i32 %a, i32 %b
  ret i32 %r
```

### CPU performance effect

Good when:

- Branch is unpredictable
- Both sides are cheap
- Avoiding misprediction is more valuable than extra work

Bad when:

- One path is expensive
- Branch is highly predictable
- Extra instructions increase critical path

### GPU performance effect

Good when:

- Branch would be divergent
- Both sides are cheap
- Avoiding EXEC-mask manipulation helps

Bad when:

- Both paths contain expensive memory operations
- Both paths increase register pressure
- Executing both paths hurts occupancy or memory bandwidth

## 8.2 `JumpThreadingPass`

### Purpose

Jump threading eliminates branches by using facts known from predecessor blocks.
## Example

### Before

```c
if (a > 0)
  x = 1;
else
  x = 2;

if (x == 1)
  foo();
else
  bar();
```

### Simplified CFG idea

```text
entry
 ├── a > 0 true  -> set x = 1 -> test x == 1 -> foo
 └── a > 0 false -> set x = 2 -> test x == 1 -> bar
```

The second branch is redundant along each predecessor.

### After jump threading

```c
if (a > 0)
  foo();
else
  bar();
```

### CPU impact

| Parameter | Impact |
|---|---|
| Dynamic branches | Reduced |
| Branch predictor | Fewer correlated branches |
| Code size | May increase if blocks are duplicated |
| I-cache | Can improve or worsen depending on duplication |
| ILP | Often improved |
| Compile-time | More CFG analysis |

### GPU impact

| Parameter | Impact |
|---|---|
| Divergence | May reduce redundant divergent branch |
| Code size | May increase |
| Register pressure | Can increase due to duplicated regions |
| Reconvergence | May simplify or complicate |
| Occupancy | Can decrease if pressure increases |

## 8.3 `DFAJumpThreadingPass`

### Purpose

Optimizes control flow that behaves like a deterministic finite automaton, especially switch-heavy loops.

Common in interpreters, parsers, and state machines.

## Example

### Before

```c
while (...) {
  switch (state) {
    case A:
      ...
      state = B;
      break;
    case B:
      ...
      state = C;
      break;
    case C:
      ...
      state = A;
      break;
  }
}
```

### After conceptual transformation

The compiler duplicates or specializes control-flow paths based on known next states.

```text
A path directly jumps to B path
B path directly jumps to C path
C path directly jumps to A path
```

### CPU impact

| Parameter | Impact |
|---|---|
| Branch prediction | Can improve by eliminating indirect/switch branches |
| I-cache | May worsen due to duplication |
| Frontend bandwidth | Can improve if fewer dispatch branches |
| Code size | Often increases |
| Interpreter performance | Often improves significantly |

### GPU impact

Usually less beneficial unless the state machine is uniform across lanes.

If state is divergent per lane:

```text
lane 0: state A
lane 1: state C
lane 2: state B
...
```

Then switch dispatch is divergent, and path specialization can increase code size without eliminating SIMT serialization.


## 8.4 `LowerSwitchPass`

### Purpose

Lowers high-level `switch` into lower-level branches or forms easier for target lowering.


## Source

```c
switch (x) {
  case 0: return a;
  case 1: return b;
  case 2: return c;
  default: return d;
}
```

### Possible lowering 1: branch chain

```llvm
%cmp0 = icmp eq i32 %x, 0
br i1 %cmp0, label %case0, label %check1

check1:
%cmp1 = icmp eq i32 %x, 1
br i1 %cmp1, label %case1, label %check2

check2:
%cmp2 = icmp eq i32 %x, 2
br i1 %cmp2, label %case2, label %default
```

### Possible lowering 2: jump table

```text
if x in range:
  indirect jump table[x]
else:
  default
```

### CPU impact

| Lowering | Good for | Bad for |
|---|---|---|
| Branch chain | Few cases, skewed probabilities | Many cases, unpredictable |
| Binary search tree | Many sparse cases | Poor locality sometimes |
| Jump table | Dense cases | Indirect branch prediction risk |
| Lookup table | Constant return values | Extra memory load |

### GPU impact

| Lowering | Impact |
|---|---|
| Branch chain | Potential repeated divergence |
| Jump table | Indirect branch may be unsupported/expensive |
| Lookup table | Memory access may diverge |
| Predicated select chain | May execute extra operations but avoid branch divergence |


## 8.5 `LoopSimplifyPass`

### Purpose

Canonicalizes loops into a form expected by other passes.

Typical properties:

- Preheader exists
- Dedicated exits
- Simplified latch structure

### Before

```text
        entry
        /   \
       v     v
    header <- latch
       |
      exit
```

Multiple outside predecessors may branch into loop header.

### After

```text
entry
  |
preheader
  |
header <--- latch
  |
exit
```

### CPU impact

| Parameter | Impact |
|---|---|
| Loop optimization | Enables LICM, vectorization, unswitching |
| Branch prediction | More canonical loop branch |
| Code size | Small increase due to preheader |
| Runtime | Usually indirect improvement |

### GPU impact

| Parameter | Impact |
|---|---|
| Structurization | Easier loop region recognition |
| Barrier correctness | Cleaner dominance/loop structure |
| Divergence analysis | More predictable loop regions |
| Occupancy | Usually no direct impact |


## 8.6 `LCSSAPass`

### Purpose

Loop-Closed SSA ensures values defined inside loops and used outside pass through exit PHIs.

### Before

```llvm
loop:
  %x = add i32 %i, 1
  br i1 %cond, label %loop, label %exit

exit:
  use %x
```

### After

```llvm
loop:
  %x = add i32 %i, 1
  br i1 %cond, label %loop, label %exit

exit:
  %x.lcssa = phi i32 [ %x, %loop ]
  use %x.lcssa
```

### CPU impact

Enables:

- Loop unswitching
- Loop rotation
- LICM
- Vectorization
- Loop deletion
- SCEV reasoning

### GPU impact

Enables:

- Structured loop lowering
- Correct PHI handling after divergent control flow
- Safer transformation around loop exits


## 8.7 `SimpleLoopUnswitchPass`

### Purpose

Moves loop-invariant condition outside the loop.


## Example

### Before

```c
for (int i = 0; i < n; ++i) {
  if (flag)
    A[i] = B[i] + 1;
  else
    A[i] = B[i] - 1;
}
```

### After

```c
if (flag) {
  for (int i = 0; i < n; ++i)
    A[i] = B[i] + 1;
} else {
  for (int i = 0; i < n; ++i)
    A[i] = B[i] - 1;
}
```

### CPU impact

| Parameter | Impact |
|---|---|
| Dynamic branches | Removes branch from loop body |
| Vectorization | Often enables better vectorization |
| I-cache | Code duplicated |
| Branch prediction | Outer branch likely cheap |
| Loop body | Cleaner and smaller per path |
| Register pressure | Usually improves inside loop |

### GPU impact

If `flag` is uniform:

| Parameter | Impact |
|---|---|
| Divergence | None; uniform branch outside loop |
| Loop body | Cleaner |
| Register pressure | Can improve |
| Occupancy | May improve if pressure reduced |

If `flag` is divergent:

```text
Different lanes choose different loop versions.
```

Then unswitching can be harmful:

| Parameter | Impact |
|---|---|
| Divergence | Whole loop becomes divergent |
| Reconvergence | More complex |
| Code size | Duplicated loop body |
| Occupancy | May decrease |
| Mask overhead | Potentially significant |


## 8.8 `LoopUnrollPass`

### Purpose

Duplicates loop body to reduce branch overhead and expose optimization opportunities.

### Before

```c
for (int i = 0; i < n; ++i)
  sum += A[i];
```

### After unroll by 4

```c
for (int i = 0; i < n; i += 4) {
  sum += A[i];
  sum += A[i + 1];
  sum += A[i + 2];
  sum += A[i + 3];
}
```

### CPU impact

| Parameter | Impact |
|---|---|
| Branch overhead | Reduced |
| ILP | Increased |
| Vectorization | Often improved |
| Register pressure | Increased |
| Code size | Increased |
| I-cache | May worsen |
| Backend scheduling | More freedom |

### GPU impact

| Parameter | Impact |
|---|---|
| Branch overhead | Reduced |
| ILP per thread | Increased |
| Register pressure | Often increased |
| Occupancy | May decrease |
| Latency hiding | Can improve or worsen |
| Code size | Increased |
| Memory coalescing | Usually unchanged unless indexing changes |

GPU-specific tradeoff:

```text
More unrolling:
  + more ILP per thread
  - more registers per thread
  - lower occupancy
```

This is one of the most important GPU control-flow tradeoffs.


# 9. Detailed GPU Pass Examples and Transformations


## 9.1 Divergence / Uniformity Analysis

### Purpose

Determine whether values and branches are uniform across lanes or divergent.


## Example

```c
int tid = threadIdx.x;

if (tid == 0)
  foo();
else
  bar();
```

`tid` is divergent because each lane has a different `threadIdx.x`.

Therefore:

```text
tid == 0 is divergent
branch is divergent
```


## Example

```c
int bid = blockIdx.x;

if (bid == 0)
  foo();
else
  bar();
```

Within a block, `blockIdx.x` is uniform.

Therefore:

```text
bid == 0 is uniform within the workgroup/wave
branch is uniform
```


## Why this matters

| Branch kind | Lowering |
|---|---|
| Uniform branch | Scalar branch, all lanes go same way |
| Divergent branch | Masked execution, reconvergence needed |


## 9.2 GPU If-Then-Else Lowering

### Source

```c
if (cond)
  x = a + b;
else
  x = a - b;
```


## Uniform condition

If `cond` is uniform:

```text
s_cbranch cond, then
s_branch else
```

All lanes go one way.

### Performance

| Parameter | Impact |
|---|---|
| Lane utilization | 100% |
| Mask overhead | None/minimal |
| Branch cost | Low |
| Register pressure | Normal |


## Divergent condition

If `cond` is divergent:

```text
save EXEC
EXEC = EXEC & cond_mask
execute then

EXEC = saved_EXEC & ~cond_mask
execute else

EXEC = saved_EXEC
```

Conceptual AMDGPU-like lowering:

```text
s_mov_b64 saved_exec, exec
v_cmpx cond
; then block under EXEC mask

s_andn2_b64 exec, saved_exec, exec
; else block under complementary mask

s_mov_b64 exec, saved_exec
```

### Performance

| Parameter | Impact |
|---|---|
| Lane utilization | Reduced |
| Dynamic instruction count | Both paths may execute |
| EXEC mask pressure | Increased |
| SGPR pressure | Increased for saved masks |
| VGPR pressure | May increase |
| Occupancy | May decrease |
| Reconvergence overhead | Added |


## 9.3 `StructurizeCFGPass`

### Purpose

Transforms arbitrary CFG into structured control-flow regions.

Important for:

- GPUs
- SPIR-V
- Shader compilers
- Targets requiring structured control flow
- SIMT reconvergence reasoning


## Unstructured CFG

```text
      A
     / \
    B   C
     \ /
      D
     / \
    E   F
     \ /
      G
```

With arbitrary jumps, early exits, and shared joins.

### Structured form

```text
if (...) {
  ...
} else {
  ...
}

if (...) {
  ...
} else {
  ...
}
```

The compiler may introduce flow variables and extra blocks.


## Example transformation idea

### Before

```llvm
A:
  br i1 %c1, label %B, label %C

B:
  br i1 %c2, label %E, label %D

C:
  br label %D

D:
  br i1 %c3, label %E, label %F

E:
  br label %G

F:
  br label %G
```

### After conceptual structurization

```llvm
A:
  %flow = ...
  br label %structured.region

structured.region:
  ; conditionally execute B/C/D/E/F using flow predicates
  br label %G
```

### GPU impact

| Parameter | Impact |
|---|---|
| Reconvergence | Easier to reason about |
| Mask handling | More regular |
| Code size | May increase |
| Register pressure | May increase due to flow variables |
| Branch count | May increase or decrease |
| Correctness | Enables legal lowering for structured targets |


## 9.4 `AMDGPUUnifyDivergentExitNodesPass`

### Purpose

AMDGPU-specific pass that ensures there is at most one divergent exiting block.

This helps `StructurizeCFGPass`, which may not handle multiple divergent exits well.


## Example

### Before

```c
if (threadIdx.x < 8)
  return 1;

if (threadIdx.x > 24)
  return 2;

return 3;
```

Multiple divergent returns.

### Conceptual after

```c
int ret;

if (threadIdx.x < 8) {
  ret = 1;
  goto unified_return;
}

if (threadIdx.x > 24) {
  ret = 2;
  goto unified_return;
}

ret = 3;

unified_return:
return ret;
```

### GPU impact

| Parameter | Impact |
|---|---|
| Structurization | Easier |
| Reconvergence | More explicit |
| PHI/register pressure | May increase |
| Code size | Slight increase |
| Correctness | Avoids problematic multi-exit divergent regions |


## 9.5 `SIAnnotateControlFlowPass`

### Purpose

AMDGPU pass that annotates control-flow constructs so later lowering can manipulate `EXEC` masks correctly.

It relies on uniformity information.


## Example

```c
if (threadIdx.x & 1)
  x = foo();
else
  x = bar();
```

The condition is divergent.

The pass helps identify regions that require mask manipulation.

Conceptual metadata / annotation:

```text
if-divergent-begin
  then region
else
  else region
if-divergent-end
```

### GPU impact

| Parameter | Impact |
|---|---|
| EXEC mask lowering | Required |
| SGPR usage | May increase for saved masks |
| VGPR usage | May increase from PHIs |
| Reconvergence | Explicit |
| Scheduling | Later passes can optimize mask operations |


## 9.6 `SILowerControlFlow`

### Purpose

Lowers AMDGPU control-flow pseudo constructs into real machine-level branches and execution-mask instructions.


## Conceptual before

```text
IF %cond
  THEN block
ELSE
  ELSE block
ENDIF
```

### Conceptual after

```text
s_mov_b64 saved_exec, exec
v_cmpx cond
s_cbranch_execz ELSE

THEN:
  ...

ELSE:
  s_andn2_b64 exec, saved_exec, exec
  s_cbranch_execz ENDIF
  ...

ENDIF:
  s_mov_b64 exec, saved_exec
```

### GPU impact

| Parameter | Impact |
|---|---|
| Dynamic instructions | Increases due to mask ops |
| SGPR pressure | Saved EXEC masks consume scalar registers |
| Branch cost | Depends on uniform/divergent behavior |
| Occupancy | Can decrease if SGPR pressure rises |
| Lane utilization | Depends on divergence |
| Correctness | Essential for SIMT execution |


## 9.7 `SIOptimizeExecMasking`

### Purpose

Optimizes redundant or inefficient EXEC mask manipulation.


## Before

```text
save exec
modify exec
restore exec

save exec
modify exec
restore exec
```

### After

```text
save exec
modify exec for combined region
restore exec
```

Or remove redundant mask operations entirely.

### GPU impact

| Parameter | Impact |
|---|---|
| Instruction count | Reduced |
| SGPR pressure | May reduce |
| Occupancy | May improve |
| Latency | Reduced |
| Branch/mask overhead | Reduced |


## 9.8 `SIWholeQuadMode`

### Purpose

Pixel shaders may need whole-quad execution for derivative operations.

Derivative instructions require a 2x2 pixel quad, even if some pixels are inactive.

Example operations:

```text
ddx
ddy
texture gradient operations
```

### Concept

Normal execution mask:

```text
EXEC = active lanes only
```

Whole-quad mode:

```text
EXEC = active lanes plus helper lanes needed for derivatives
```

### GPU impact

| Parameter | Impact |
|---|---|
| Lane utilization | More lanes may execute |
| Correctness | Required for derivatives |
| Instruction count | Extra mask operations |
| Register pressure | May increase |
| Memory traffic | May increase due to helper lanes |
| Performance | Can decrease but required semantically |


# 10. Control Flow and Architectural Parameters


## 10.1 Branch Prediction

Primarily CPU concern.

### Transformation impact

| Transform | Branch prediction effect |
|---|---|
| SimplifyCFG | Removes trivial branches |
| JumpThreading | Removes correlated redundant branches |
| LoopUnswitch | Moves branch outside hot loop |
| IfConversion | Removes branch but may add work |
| MachineBlockPlacement | Improves fallthrough and prediction layout |
| TailDuplication | Removes join branches at cost of code size |

### CPU example

```c
if (likely(x > 0))
  hot();
else
  cold();
```

A CPU compiler may lay out:

```text
entry:
  branch if x <= 0 to cold
hot:
  fallthrough fast path
cold:
  out-of-line
```


## 10.2 Pipeline Flush

CPU misprediction causes pipeline flush.

| Transform | Flush impact |
|---|---|
| If-conversion | Eliminates potential flush |
| Jump threading | Removes branch |
| Block placement | Improves predicted path |
| Loop rotation | Places loop latch in predictable form |

GPU does not have the same branch predictor/mispredict model for SIMT branches.


## 10.3 Instruction Cache

Both CPU and GPU care about code size, but CPUs are often more frontend sensitive.

| Transform | Code size impact |
|---|---|
| Loop unrolling | Increases |
| Loop unswitching | Increases |
| Tail duplication | Increases |
| Jump threading | Can increase |
| SimplifyCFG | Often decreases |
| StructurizeCFG | Can increase |
| If-conversion | Can increase or decrease |

CPU risk:

```text
More code → I-cache misses → frontend stalls
```

GPU risk:

```text
More code → instruction cache pressure → lower throughput
```


## 10.4 Register Pressure

Very important on both, but especially important on GPUs.

### CPU

```text
More live values → spills to stack → memory latency
```

### GPU

```text
More registers per thread → fewer resident waves/warps → lower occupancy
```

Example:

```c
if (cond) {
  many temporaries...
} else {
  many other temporaries...
}
```

If-conversion may make temporaries from both paths live simultaneously.

### CPU effect

Can cause spills.

### GPU effect

Can reduce occupancy significantly.


## 10.5 Occupancy

GPU-specific.

Occupancy depends on:

- Registers per thread
- Shared memory per block
- Threads per block
- Hardware wave slots
- Barrier usage
- Target architecture

Control-flow transforms affect occupancy by changing:

- Register pressure
- Code shape
- PHI liveness
- Mask stack usage
- Spills
- Shared memory use indirectly

Example:

```text
Before unroll:
  32 VGPRs/thread
  occupancy = 8 waves/CU

After unroll:
  64 VGPRs/thread
  occupancy = 4 waves/CU
```

Even if unrolling reduces branches, performance may worsen due to reduced latency hiding.


## 10.6 Lane Utilization

GPU-specific.

A divergent branch:

```c
if (threadIdx.x % 2)
  A();
else
  B();
```

For warp size 32:

```text
16 lanes active for A
16 lanes inactive

then

16 lanes active for B
16 lanes inactive
```

Effective utilization may be roughly 50%, ignoring overhead.


## 10.7 Memory Coalescing

Control flow affects which lanes perform memory operations.

Example:

```c
if (threadIdx.x % 2 == 0)
  x = A[threadIdx.x];
```

Only even lanes load.

Memory coalescing may be less efficient than all lanes loading contiguous addresses.

### Transform interaction

If-conversion may convert to predicated loads.

But predicated memory is tricky:

```text
Executing masked loads may still consume memory pipeline resources.
Speculating invalid loads may be illegal.
```


## 10.8 Barriers

GPU barriers are control-flow-sensitive.

Example:

```c
if (threadIdx.x < 16) {
  __syncthreads();
}
```

This is generally illegal or dangerous if not all threads in the block reach the barrier uniformly.

Compiler must reason about:

- Uniformity
- Dominance
- Post-dominance
- Structured regions
- Barrier convergence

Control-flow transforms must preserve barrier semantics.


# 11. CPU vs GPU Transformation Trade-Off Matrix

| Transformation | CPU benefit | CPU risk | GPU benefit | GPU risk |
|---|---|---|---|---|
| SimplifyCFG | Fewer branches, simpler CFG | Over-speculation | Fewer divergent paths | May hide structure useful to GPU |
| JumpThreading | Removes redundant branches | Code duplication | Removes redundant divergent tests | More code/register pressure |
| If-conversion | Avoids mispredicts | Executes both sides | Avoids divergence/mask ops | Executes both sides, raises VGPRs |
| LoopUnswitch | Removes inner branch | Code bloat | Good for uniform branch | Bad for divergent branch |
| LoopUnroll | Reduces branch overhead, exposes ILP | I-cache/register pressure | More ILP per thread | Lower occupancy |
| TailDuplication | Better layout, fewer jumps | Code bloat | Can reduce joins | Can complicate reconvergence |
| StructurizeCFG | Rare for CPU | May pessimize CPU CFG | Required/useful for GPU | Extra flow vars, code growth |
| LowerSwitch | Enables jump tables | Indirect branch risk | Simpler lowering | Divergent switch expensive |
| BlockPlacement | Better I-cache/fallthrough | Profile sensitivity | Some benefit | Less important than divergence |
| Exec-mask optimization | N/A | N/A | Reduces mask overhead | Target-specific complexity |


# 12. Case Study 1: If-Then-Else


## Source

```c
int f(int x, int y, bool cond) {
  if (cond)
    return x + y;
  else
    return x - y;
}
```


## CPU Option A: Branch

```asm
test cond
je else
add x, y
jmp done
else:
sub x, y
done:
```

### Good when

- `cond` is predictable
- One path is expensive
- Avoiding extra computation matters

### Bad when

- `cond` is unpredictable
- Misprediction penalty is high


## CPU Option B: Select / CMOV

```asm
a = x + y
b = x - y
r = cond ? a : b
```

### Good when

- Both sides cheap
- Condition unpredictable
- Avoiding branch is profitable

### Bad when

- Both paths expensive
- Extra work increases critical path


## GPU Option A: Uniform Branch

If `cond` is uniform:

```text
all lanes take same path
```

Efficient.


## GPU Option B: Divergent Branch

If `cond` varies per lane:

```text
execute then under mask
execute else under mask
```

Potentially expensive.


## GPU Option C: Predicated Select

```text
compute both
select per lane
```

Good for cheap arithmetic.

Bad for expensive memory or function calls.


# 13. Case Study 2: Switch Lowering


## Source

```c
int classify(int x) {
  switch (x) {
    case 0: return 10;
    case 1: return 20;
    case 2: return 30;
    case 3: return 40;
    default: return -1;
  }
}
```


## CPU Lowering: Lookup Table

```c
if ((unsigned)x <= 3)
  return table[x];
else
  return -1;
```

### CPU benefits

- Removes multiple branches
- Good for dense switches
- Predictable bounds check

### CPU risks

- Extra memory load
- Cache miss if table cold


## GPU Lowering: Lookup Table

If `x` is divergent:

```text
each lane may load table[x_lane]
```

If values vary across lanes, memory access may still be coalesced if table small/cacheable.

### GPU benefits

- Avoids divergent switch chain
- Uses constant memory efficiently if uniform/broadcast

### GPU risks

- Divergent memory access
- Extra memory instruction
- Potential constant-cache serialization


# 14. Case Study 3: Loop Unswitching


## Source

```c
for (int i = 0; i < n; ++i) {
  if (mode)
    A[i] = B[i] * 2;
  else
    A[i] = B[i] + 2;
}
```


## CPU

If `mode` is loop-invariant:

```c
if (mode) {
  for (...) A[i] = B[i] * 2;
} else {
  for (...) A[i] = B[i] + 2;
}
```

### CPU impact

| Metric | Effect |
|---|---|
| Inner loop branches | Reduced |
| Vectorization | Improved |
| Code size | Increased |
| I-cache | Potentially worse |
| Branch prediction | Usually improved |


## GPU

### If `mode` is uniform

Good transformation.

```text
All lanes execute same loop version.
```

### If `mode` is divergent

Potentially bad.

```text
Some lanes enter first loop clone.
Some lanes enter second loop clone.
The whole loop body may be serialized.
```

### GPU impact

| Metric | Uniform mode | Divergent mode |
|---|---|---|
| Divergence | Low | High |
| Code size | Higher | Higher |
| Occupancy | Maybe same | May decrease |
| Lane utilization | High | Low |
| Mask overhead | Low | High |


# 15. Case Study 4: Loop Unrolling


## Source

```c
for (int i = 0; i < n; ++i) {
  acc += A[i] * B[i];
}
```


## CPU Unroll by 4

```c
for (int i = 0; i < n; i += 4) {
  acc0 += A[i] * B[i];
  acc1 += A[i+1] * B[i+1];
  acc2 += A[i+2] * B[i+2];
  acc3 += A[i+3] * B[i+3];
}
acc = acc0 + acc1 + acc2 + acc3;
```

### CPU benefits

- Reduced branch overhead
- More ILP
- Better vectorization
- Better scheduling

### CPU risks

- More registers
- More code
- I-cache pressure


## GPU Unroll by 4

Each thread performs more work.

### GPU benefits

- More ILP per lane
- Better memory latency hiding within thread
- Fewer loop branches

### GPU risks

- More VGPRs
- Lower occupancy
- Potential spills to local memory
- Code size increase

### Key GPU question

```text
Does increased ILP compensate for reduced occupancy?
```


# 16. Case Study 5: Divergent Early Exit


## Source

```c
if (threadIdx.x < 8)
  return;

do_work();
```


## Naive view

Only some lanes return.

But a warp/wave cannot simply let lanes independently disappear without mask handling.


## GPU conceptual lowering

```text
saved_exec = exec

cond_mask = threadIdx.x < 8
exec = exec & ~cond_mask

if exec == 0:
  jump function_exit

do_work under remaining lanes

function_exit:
exec = saved_exec
return
```


## Compiler issues

| Issue | Explanation |
|---|---|
| Divergent exit | Some lanes exit, others continue |
| Reconvergence | Need safe point where wave is whole again |
| PHIs | Return values may differ |
| Structurization | Multiple exits are hard |
| Barriers | Early exit before barrier can be illegal |


## AMDGPU pass relevance

| Pass | Role |
|---|---|
| `AMDGPUUnifyDivergentExitNodesPass` | Creates unified divergent exit |
| `StructurizeCFGPass` | Structures region |
| `SIAnnotateControlFlowPass` | Marks divergent control |
| `SILowerControlFlow` | Emits EXEC-mask logic |


# 17. Case Study 6: Irreducible Control Flow


## Irreducible CFG

An irreducible CFG has a loop with multiple entries.

```text
     A
    / \
   v   v
   B <- C
   |    ^
   v    |
   D ---+
```

This is hard for many loop analyses and especially problematic for structured GPU targets.


## CPU handling

CPUs can execute arbitrary branches.

The compiler may still normalize irreducible CFG because:

- Loop optimizations need natural loops
- Profile analysis is easier
- Register allocation and block layout improve


## GPU handling

GPU compilers often need reducible/structured control flow.

Passes like:

- `FixIrreduciblePass`
- `StructurizeCFGPass`

may transform the CFG using extra dispatch variables or duplicated blocks.


## Performance impact

| Metric | Impact |
|---|---|
| Code size | May increase |
| Register pressure | May increase |
| Divergence | May become more explicit |
| Correctness | Required for structured targets |
| CPU performance | Can be neutral or negative |
| GPU performance | Often necessary despite overhead |


# 18. CPU Machine-Level Control Flow


## 18.1 `MachineBlockPlacementPass`

### Purpose

Orders machine basic blocks for better fallthrough and locality.


## Example

Profile says:

```text
entry -> hot_path: 95%
entry -> cold_path: 5%
```

### Before

```text
entry
cold_path
hot_path
exit
```

### After

```text
entry
hot_path
exit
cold_path
```

### CPU impact

| Parameter | Impact |
|---|---|
| Fallthrough | Improved |
| Branch prediction | Improved |
| I-cache locality | Improved |
| TLB/cache | Better hot-code clustering |

### GPU impact

Less central than CPU, but instruction locality still matters.


## 18.2 `BranchFolderPass`

### Purpose

Removes redundant branches and merges equivalent branch tails.

### Example

Before:

```asm
B1:
  jmp B3

B2:
  jmp B3

B3:
  ret
```

After:

```asm
B1:
B2:
B3:
  ret
```

### CPU impact

- Fewer jumps
- Smaller code
- Better layout


## 18.3 `TailDuplicatePass`

### Purpose

Duplicates small block tails to remove branches or improve layout.

### Before

```text
A -> C
B -> C
C -> D
```

### After

```text
A -> C_clone_for_A -> D
B -> C_clone_for_B -> D
```

### CPU impact

| Benefit | Risk |
|---|---|
| Fewer branches | Code size increase |
| Better scheduling | I-cache pressure |
| Better fallthrough | More compile-time |

### GPU impact

Can reduce some joins, but may complicate reconvergence and increase code/register pressure.


## 18.4 `IfConverterPass`

### Purpose

Converts branches into predicated instructions.

### Before

```asm
cmp r0, #0
beq L1
add r1, r2, r3
b L2
L1:
sub r1, r2, r3
L2:
```

### After

```asm
cmp r0, #0
add.ne r1, r2, r3
sub.eq r1, r2, r3
```

### CPU impact

Good for predicated architectures or unpredictable branches.

Bad if predicated body is too large.


# 19. GPU Machine-Level Control Flow


## 19.1 Execution Mask

GPU divergent control flow is usually implemented with an execution mask.

For a wave of 8 lanes:

```text
EXEC = 11111111
```

If condition true for lanes 0, 2, 4, 6:

```text
COND = 01010101
EXEC = EXEC & COND = 01010101
```

Only those lanes execute.


## 19.2 Divergent If Lowering

```c
if (cond)
  then_body();
else
  else_body();
```

Conceptual lowering:

```text
saved_exec = exec

then_mask = exec & cond
exec = then_mask
if exec != 0:
  then_body()

else_mask = saved_exec & ~cond
exec = else_mask
if exec != 0:
  else_body()

exec = saved_exec
```


## 19.3 Nested Divergence

```c
if (a) {
  if (b) {
    work1();
  } else {
    work2();
  }
} else {
  work3();
}
```

Potential mask stack:

```text
save exec for a
  exec &= a
  save exec for b
    exec &= b
    work1
    exec = saved_b & ~b
    work2
  restore b
exec = saved_a & ~a
work3
restore a
```

### GPU impact

| Parameter | Impact |
|---|---|
| Mask operations | Increase with nesting |
| SGPR pressure | Saved masks consume scalar registers |
| Code size | Increased |
| Lane utilization | Lower if divergent |
| Occupancy | May decrease |


## 19.4 Uniform Branch Optimization

If compiler proves a branch is uniform:

```c
if (blockIdx.x == 0)
  foo();
else
  bar();
```

It can lower to scalar control flow:

```text
s_cmp blockIdx.x, 0
s_cbranch_scc0 else
foo
s_branch end
else:
bar
end:
```

No per-lane EXEC mask split needed.

This is why `UniformityAnalysis` is crucial.


# 20. AMDGPU Control-Flow Pipeline Example

A simplified AMDGPU control-flow-related sequence:

```text
LowerSwitch
FlattenCFG
Sinking
AMDGPUUnifyDivergentExitNodes
FixIrreducible
UnifyLoopExits
StructurizeCFG
AMDGPUAnnotateUniformValues
SIAnnotateControlFlow
AMDGPURewriteUndefForPHI
LCSSA
UniformityInfoAnalysis
Instruction Selection
SILowerControlFlow
SIOptimizeExecMaskingPreRA
Register Allocation
SIOptimizeExecMasking
SILateBranchLowering
```


## Why this sequence exists

### `LowerSwitch`

Switches are complex. Lower them before GPU-specific CFG handling.

### `FlattenCFG`

Simplifies some branch structures before structurization.

### `AMDGPUUnifyDivergentExitNodes`

Multiple divergent exits are hard to structurize.

### `FixIrreducible`

Irreducible CFG is problematic for structured SIMT lowering.

### `UnifyLoopExits`

Loops with multiple exits are harder to structure.

### `StructurizeCFG`

Creates structured regions.

### `AMDGPUAnnotateUniformValues`

Marks uniform values/branches to avoid unnecessary EXEC manipulation.

### `SIAnnotateControlFlow`

Annotates AMDGPU control-flow regions.

### `SILowerControlFlow`

Turns annotations/pseudo-control-flow into machine instructions.

### `SIOptimizeExecMasking`

Removes redundant mask operations.


# 21. NVPTX Control-Flow Pipeline Notes

NVPTX differs from AMDGPU because LLVM emits PTX, and final SASS generation is performed by NVIDIA tooling.

Important concepts:

| Concept | Description |
|---|---|
| Predicates | PTX uses predicate registers for conditional execution |
| Branches | PTX has conditional branches |
| Reconvergence | Some handled by PTX/SASS-level mechanisms |
| Divergence | Still semantically important |
| Uniformity | Useful for avoiding expensive divergent control |
| Kernel ABI | Control flow begins at kernel entry |

Example PTX-like control flow:

```ptx
setp.ne.s32 %p1, %r1, 0;
@%p1 bra THEN;
bra ELSE;

THEN:
  ...
  bra END;

ELSE:
  ...

END:
```

Predicated instruction:

```ptx
@%p1 add.s32 %r2, %r3, %r4;
```

CPU-like branch layout still matters somewhat, but divergence dominates GPU behavior.


# 22. SPIR-V and Structured Control Flow

SPIR-V often requires structured control flow.

Example SPIR-V concepts:

- `OpSelectionMerge`
- `OpLoopMerge`
- `OpBranchConditional`
- `OpBranch`

A structured if:

```text
OpSelectionMerge %merge None
OpBranchConditional %cond %then %else

%then:
  ...
  OpBranch %merge

%else:
  ...
  OpBranch %merge

%merge:
```

A loop:

```text
OpLoopMerge %merge %continue None
OpBranchConditional %cond %body %merge
```

This is why structurization is central for SPIR-V-like paths.


# 23. MLIR Control-Flow Lowering View

MLIR represents control flow at multiple abstraction levels.

## Structured Control Flow

```mlir
scf.if %cond {
  ...
} else {
  ...
}
```

```mlir
scf.for %i = %lb to %ub step %step {
  ...
}
```

## CFG Control Flow

```mlir
cf.cond_br %cond, ^then, ^else
```

## GPU Dialect

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%gx, %gy, %gz)
           threads(%tx, %ty, %tz) in (%sx, %sy, %sz) {
  ...
}
```

## Lowering paths

```text
linalg/tensor
  -> scf/affine/vector
  -> gpu
  -> nvvm / rocdl / spirv
  -> LLVM IR / binary
```

Control-flow choices at high levels affect:

- Tiling
- Fusion
- Vectorization
- Mapping to threads
- Barrier placement
- Shared memory lifetime
- Divergence


# 24. Performance Modeling Checklist

## CPU Control-Flow Checklist
1. Is the branch predictable?
2. Is the branch inside a hot loop?
3. Is one path much hotter than the other?
4. Is if-conversion profitable?
5. Will code duplication hurt I-cache?
6. Does transformation enable vectorization?
7. Does transformation increase register pressure?
8. Does switch lowering produce a jump table?
9. Are indirect branches expensive on this CPU?
10. Does block layout match profile data?

## GPU Control-Flow Checklist

1. Is the branch uniform or divergent?
2. What fraction of lanes take each path?
3. Are paths balanced or highly asymmetric?
4. Are there memory operations inside divergent paths?
5. Does if-conversion execute unsafe/speculative memory?
6. Does transformation increase VGPR pressure?
7. Does it reduce occupancy?
8. Does it introduce more SGPR mask saves?
9. Are barriers control-dependent?
10. Does structurization add flow variables?
11. Does loop unrolling reduce occupancy?
12. Are early exits divergent?
13. Are PHIs after divergent regions uniform or divergent?
14. Are memory accesses still coalesced?
15. Does target support cheap uniform branches?



## Key things to inspect in IR

Look for:

```llvm
br i1
switch
select
phi
llvm.experimental.convergence
convergent calls
```

For GPU:

```llvm
addrspace
amdgpu intrinsics
nvvm intrinsics
thread/block ID intrinsics
convergent operations
barriers
```


## Key things to inspect in machine code

CPU:

```text
conditional branches
cmov/select
jump tables
fallthrough layout
loop latch
unrolled bodies
```

GPU:

```text
EXEC mask manipulation
scalar branches
vector compares
predicate registers
reconvergence markers
barrier instructions
saved mask registers
```
