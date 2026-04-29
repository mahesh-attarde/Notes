# GPU control-flow transformation using LLVM Divergence / Uniformity Analysis**

## 1. Motivation

GPU execution is typically **SPMD/SIMT**:

- One logical program instance is executed by many lanes/threads.
- Threads are grouped into execution units:
  - NVIDIA: warp
  - AMDGPU: wavefront / wave
  - SPIR-V/OpenCL: subgroup/workgroup concepts
- A single instruction stream is shared by many lanes.
- Per-lane predicates/masks determine which lanes are active.

A branch is cheap when all lanes agree on the direction. It is expensive when lanes disagree.

That distinction is the core reason for **divergence analysis** and **uniformity analysis**.

A value is:

- **Uniform** if it has the same value for all lanes in the relevant execution group.
- **Divergent** if it may differ between lanes.

A branch is:

- **Uniform control flow** if its condition is uniform.
- **Divergent control flow** if its condition is divergent.

For GPU compilers, this matters because divergent control flow can cause:

- mask stack manipulation,
- reconvergence overhead,
- serialized execution of paths,
- inhibited instruction scheduling,
- inhibited memory coalescing,
- poor occupancy or register pressure behavior,
- invalid transformations around convergent operations.

LLVM contains a generic uniformity framework used by IR and Machine IR. Relevant files include:

- [`llvm/include/llvm/Analysis/UniformityAnalysis.h`](https://github.com/llvm/llvm-project/blob/5249e5527f48e1dd6b13c3dabed4992b299bfcb9/llvm/include/llvm/Analysis/UniformityAnalysis.h)
- [`llvm/lib/Analysis/UniformityAnalysis.cpp`](https://github.com/llvm/llvm-project/blob/5249e5527f48e1dd6b13c3dabed4992b299bfcb9/llvm/lib/Analysis/UniformityAnalysis.cpp)
- [`llvm/include/llvm/ADT/GenericUniformityInfo.h`](https://github.com/llvm/llvm-project/blob/5249e5527f48e1dd6b13c3dabed4992b299bfcb9/llvm/include/llvm/ADT/GenericUniformityInfo.h)
- [`llvm/include/llvm/ADT/GenericUniformityImpl.h`](https://github.com/llvm/llvm-project/blob/5249e5527f48e1dd6b13c3dabed4992b299bfcb9/llvm/include/llvm/ADT/GenericUniformityImpl.h)
- [`llvm/include/llvm/CodeGen/MachineUniformityAnalysis.h`](https://github.com/llvm/llvm-project/blob/5249e5527f48e1dd6b13c3dabed4992b299bfcb9/llvm/include/llvm/CodeGen/MachineUniformityAnalysis.h)

The key practical point:

> Control-flow transformations on GPU should distinguish uniform branches from divergent branches. Treating all branches equally is often suboptimal or incorrect.
# 2. Uniformity vs Divergence

## 2.1 Uniform values

Examples of usually-uniform values:

- Kernel scalar arguments marked or known uniform by target/frontend.
- Constants.
- Function-scope invariant values.
- Workgroup-level constants.
- Values loaded from uniform metadata or scalar memory, depending on target.
- Results of operations whose operands are uniform.
- Some target-specific intrinsics, e.g. “read grid size”, “read block id”, depending on target semantics.

## 2.2 Divergent values

Examples of usually-divergent values:

- Thread/lane ID.
- Subgroup lane ID.
- Per-thread pointer arithmetic based on lane ID.
- Loads from per-thread memory.
- Results of arithmetic involving divergent operands.
- PHI nodes selected by divergent control flow.
- Values defined in loops with divergent exits, depending on temporal divergence.

Example:

```llvm name=basic-divergence.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @k(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %is_even = icmp eq i32 (and i32 %tid, 1), 0
  br i1 %is_even, label %then, label %else

then:
  store i32 1, ptr addrspace(1) %out
  br label %merge

else:
  store i32 2, ptr addrspace(1) %out
  br label %merge

merge:
  ret void
}
```

Here `%tid` is divergent, `%is_even` is divergent, and the branch is divergent.
# 3. What LLVM Uniformity Analysis Computes

LLVM’s `UniformityInfo` answers questions such as:

- Is this value divergent?
- Is this instruction divergent?
- Is this use divergent?
- Does this block have a divergent terminator?
- Are there temporally divergent values due to loop exits?

The public interface is exposed through `GenericUniformityInfo`.

Important query shape:

```cpp name=uniformity-api.cpp
bool isDivergent(ConstValueRefT V) const;
bool isUniform(ConstValueRefT V) const;
bool isDivergent(const InstructionT *I) const;
bool isUniform(const InstructionT *I) const;
bool isDivergentUse(const UseT &U) const;
bool hasDivergentTerminator(const BlockT &B);
```

In LLVM IR, `UniformityInfo` is the `SSAContext` specialization:

```cpp name=uniformity-info-typedef.cpp
using UniformityInfo = GenericUniformityInfo<SSAContext>;
```

The IR pass is:

```cpp name=uniformity-pass.cpp
class UniformityInfoAnalysis
    : public AnalysisInfoMixin<UniformityInfoAnalysis> {
public:
  using Result = UniformityInfo;
  UniformityInfo run(Function &F, FunctionAnalysisManager &);
};
```

LLVM only computes the real analysis for targets that report branch divergence through TTI:

```cpp name=tti-branch-divergence.cpp
bool TargetTransformInfo::hasBranchDivergence(const Function *F) const;
```

For example:

- NVPTX reports branch divergence as true.
- AMDGPU reports branch divergence unless the function executes in single-lane mode.
- Generic/default CPU TTI reports false.

This distinction is important: on CPUs, LLVM can often ignore GPU-style uniformity.
# 4. Sources of Divergence

LLVM’s generic uniformity framework starts by assuming values are uniform and then propagates divergence from target/frontend-defined sources.

At IR level, the initialization path uses `TargetTransformInfo::getValueUniformity`.

Conceptually:

```cpp name=conceptual-uniformity-init.cpp
for each argument:
  if TTI says NeverUniform:
    seed divergence
  else:
    mark uniform

for each instruction:
  switch TTI.getValueUniformity(I):
    case AlwaysUniform:
      mark always uniform
    case NeverUniform:
      seed divergence
    case Custom:
      mark as conditionally uniform depending on operand uniformity
    case Default:
      initially mark uniform
```

Target-specific logic decides that things like lane ID are divergent.

Example:

```llvm name=source-of-divergence.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @source(ptr addrspace(1) %out) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %x = add i32 %lane, 42
  store i32 %x, ptr addrspace(1) %out
  ret void
}
```

Divergence propagation:

```text name=source-of-divergence-flow.txt
%lane is divergent
  -> %x is divergent because it uses %lane
    -> store address or stored value may be divergent depending on operands
```
# 5. Data Dependence vs Sync Dependence

Divergence is not only propagated through ordinary SSA use-def edges.

It also propagates through **control flow**.

Consider:

```llvm name=sync-dependence.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @sync_dep(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp slt i32 %tid, 10
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %a = phi i32 [ 0, %then ], [ 1, %else ]
  store i32 %a, ptr addrspace(1) %out
  ret void
}
```

`%a` has no direct data dependence on `%tid`.

But `%a` is divergent because the branch condition is divergent. Different lanes can arrive at `%merge` from different predecessor blocks and choose different incoming PHI values.

This is called **sync dependence** or **control-induced divergence**.

LLVM’s `GenericSyncDependenceAnalysis` models this.

The high-level idea:

```text name=sync-dependence-concept.txt
Given a divergent branch B:

  if two disjoint paths from B reach a join block J,
  then PHI nodes in J may become divergent.

For loops/cycles:
  if divergent paths affect cycle exits,
  values defined inside the cycle may become temporally divergent
  when used outside the cycle.
```
# 6. Divergent Branches and Join Blocks

A classic CFG:

```text name=diamond-cfg.txt
entry:
  br i1 %div_cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %x = phi i32 [ 1, %then ], [ 2, %else ]
```

If `%div_cond` is divergent:

- `entry` has a divergent terminator.
- `merge` is a join block.
- PHI nodes in `merge` may be divergent.

If `%div_cond` is uniform:

- All lanes take the same edge.
- `%x` may be uniform if incoming values are uniform.
- The branch can usually be treated as normal scalar control flow.

This is the first major opportunity for control-flow transformation.
# 7. Control-Flow Transformations Driven by Uniformity

There are several transformation families.

## 7.1 Avoid if-converting uniform branches

On CPUs, branch-to-select conversion is often profitable for unpredictable branches.

On GPUs, a uniform branch is often very cheap because all lanes go the same way. Converting it to predicated/select form can make both sides execute unnecessarily.

Before:

```llvm name=uniform-branch-before.ll
define amdgpu_kernel void @uniform_branch(ptr addrspace(1) %out, i32 %n) {
entry:
  %cond = icmp sgt i32 %n, 0
  br i1 %cond, label %then, label %else

then:
  store i32 11, ptr addrspace(1) %out
  br label %merge

else:
  store i32 22, ptr addrspace(1) %out
  br label %merge

merge:
  ret void
}
```

If `%n` is known uniform, then `%cond` is uniform.

Bad transformation:

```llvm name=uniform-branch-bad-ifconvert.ll
define amdgpu_kernel void @uniform_branch_bad(ptr addrspace(1) %out, i32 %n) {
entry:
  %cond = icmp sgt i32 %n, 0
  %v = select i1 %cond, i32 11, i32 22
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

This is not always wrong, but it may be undesirable:

- select value is still uniform,
- but if the original branches contained expensive code, if-conversion could execute both sides,
- uniform branch allows the whole wave to skip one side.

Transformation rule:

```text name=uniform-ifconversion-rule.txt
If branch condition is uniform:
  Prefer preserving control flow unless:
    - both sides are tiny,
    - transformation exposes major simplification,
    - target cost model says select/predication is cheaper.
```

## 7.2 If-convert divergent branches when profitable

Divergent branches can serialize paths. For small side-effect-free regions, replacing control flow with select can be profitable.

Before:

```llvm name=divergent-if-before.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @div_if(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, 32
  br i1 %cond, label %then, label %else

then:
  %a = add i32 %tid, 10
  br label %merge

else:
  %b = sub i32 %tid, 10
  br label %merge

merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

After:

```llvm name=divergent-if-after-select.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @div_if_select(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, 32
  %a = add i32 %tid, 10
  %b = sub i32 %tid, 10
  %v = select i1 %cond, i32 %a, i32 %b
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

This removes divergent control flow.

But this is only valid/profitable when:

- both regions are safe to speculate,
- instructions have no side effects,
- memory operations are safe or can be predicated,
- no convergent operations are duplicated/moved illegally,
- register pressure does not become too high,
- region size is small enough.

LLVM has a `SpeculativeExecutionPass` aimed partly at GPU targets. Its header says it hoists instructions to enable speculative execution on targets where branches are expensive and can be gated by `TargetTransformInfo::hasBranchDivergence()`.

Relevant file:

- [`llvm/include/llvm/Transforms/Scalar/SpeculativeExecution.h`](https://github.com/llvm/llvm-project/blob/5249e5527f48e1dd6b13c3dabed4992b299bfcb9/llvm/include/llvm/Transforms/Scalar/SpeculativeExecution.h)

## 7.3 Hoist uniform computations out of divergent regions

Suppose a divergent branch contains a computation that is uniform and identical on both paths.

Before:

```llvm name=hoist-uniform-before.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @hoist_uniform(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, 32
  br i1 %cond, label %then, label %else

then:
  %u1 = mul i32 %n, 4
  %a = add i32 %u1, %tid
  br label %merge

else:
  %u2 = mul i32 %n, 4
  %b = sub i32 %u2, %tid
  br label %merge

merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

If `%n` is uniform, `%u1` and `%u2` are uniform and equivalent.

After:

```llvm name=hoist-uniform-after.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @hoist_uniform_after(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %u = mul i32 %n, 4
  %cond = icmp ult i32 %tid, 32
  br i1 %cond, label %then, label %else

then:
  %a = add i32 %u, %tid
  br label %merge

else:
  %b = sub i32 %u, %tid
  br label %merge

merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

This can reduce duplicate scalar work and may improve scalar-register usage on AMDGPU.

Transformation rule:

```text name=uniform-hoisting-rule.txt
Inside divergent control:
  If instruction is uniform,
  and operands dominate the target hoist point,
  and instruction is speculatable or guaranteed executed on all paths,
  then hoist to nearest uniform dominator region.
```

## 7.4 Sink divergent computations into divergent regions

Sometimes divergent computation is only needed on one side of a branch.

Before:

```llvm name=sink-divergent-before.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @sink_div(ptr addrspace(1) %out, i32 %flag) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %exp = mul i32 %tid, 17
  %cond = icmp ne i32 %flag, 0
  br i1 %cond, label %then, label %merge

then:
  store i32 %exp, ptr addrspace(1) %out
  br label %merge

merge:
  ret void
}
```

If `%flag` is uniform, and the `then` block is skipped by the whole wave when false, computing `%exp` before the branch is wasted.

After:

```llvm name=sink-divergent-after.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @sink_div_after(ptr addrspace(1) %out, i32 %flag) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ne i32 %flag, 0
  br i1 %cond, label %then, label %merge

then:
  %exp = mul i32 %tid, 17
  store i32 %exp, ptr addrspace(1) %out
  br label %merge

merge:
  ret void
}
```

This is especially useful when:

- branch condition is uniform,
- computation is divergent/expensive,
- computation is only needed in one branch,
- sinking does not increase dynamic executions.

## 7.5 Split uniform and divergent parts of a condition

A branch condition can combine uniform and divergent predicates.

Before:

```llvm name=mixed-condition-before.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @mixed(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %ucond = icmp sgt i32 %n, 0
  %dcond = icmp ult i32 %tid, 32
  %cond = and i1 %ucond, %dcond
  br i1 %cond, label %then, label %else

then:
  store i32 1, ptr addrspace(1) %out
  br label %merge

else:
  store i32 0, ptr addrspace(1) %out
  br label %merge

merge:
  ret void
}
```

If `%ucond` is uniform and `%dcond` is divergent, the combined `%cond` is divergent. But preserving the uniform guard separately may reduce work.

After:

```llvm name=mixed-condition-after.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @mixed_split(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %ucond = icmp sgt i32 %n, 0
  br i1 %ucond, label %uniform_true, label %else

uniform_true:
  %dcond = icmp ult i32 %tid, 32
  br i1 %dcond, label %then, label %else

then:
  store i32 1, ptr addrspace(1) %out
  br label %merge

else:
  store i32 0, ptr addrspace(1) %out
  br label %merge

merge:
  ret void
}
```

This can help because:

- if `%ucond` is false, the whole wave skips the divergent branch,
- expensive divergent computation can be avoided,
- reconvergence pressure is reduced.

This resembles boolean condition factoring.

Transformation rule:

```text name=condition-factorization-rule.txt
Given condition C = U && D:
  where U is uniform and D is divergent,
  transform:
    br (U && D), T, F
  into:
    br U, check_D, F
    check_D:
      br D, T, F

Given condition C = U || D:
  transform:
    br U, T, check_D
    check_D:
      br D, T, F
```

Profitability depends on code size, branch cost, and whether `D` is expensive or has side effects.
# 8. Divergent Control-Flow Structurization

Many GPU backends prefer or require structured control flow.

For example, AMDGPU pipelines include passes such as:

- `StructurizeCFG`
- `FixIrreducible`
- `UnifyLoopExits`
- `Unify divergent function exit nodes`

The pipeline snippet found in `llvm/test/CodeGen/AMDGPU/llc-pipeline.ll` shows `Cycle Info Analysis`, `Uniformity Analysis`, and AMDGPU late IR optimizations near CFG-related passes.

A common GPU compiler strategy:

```text name=gpu-cfg-pipeline.txt
1. Simplify CFG.
2. Lower switches.
3. Flatten simple CFGs.
4. Sink/hoist code using uniformity.
5. Compute CycleInfo and UniformityInfo.
6. Transform divergent regions.
7. Structurize irreducible or unstructured control flow.
8. Preserve convergent semantics.
9. Lower to target-specific mask/reconvergence operations.
```
# 9. Region If-Conversion Using Uniformity

A useful transformation pass can be described as:

> Convert small divergent single-entry/single-exit regions into straight-line predicated code, while preserving uniform branches.

## 9.1 Candidate shape

```text name=ifconversion-candidate-cfg.txt
      entry
        |
        v
      branch
      /    \
   then    else
      \    /
       merge
```

Requirements:

- Branch condition is divergent.
- `then` and `else` are side-effect-free or safely predicatable.
- Merge block has PHIs only from `then`/`else`.
- No unsafe memory operations unless target supports predication/masking.
- No convergent operations that would be duplicated or speculated illegally.
- No exceptional control flow.
- Region size within threshold.
- Estimated register pressure acceptable.

## 9.2 Before

```llvm name=region-ifconvert-before.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @region_ifcvt(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp slt i32 %tid, 16
  br i1 %cond, label %then, label %else

then:
  %t0 = add i32 %tid, 1
  %t1 = mul i32 %t0, 3
  br label %merge

else:
  %e0 = sub i32 %tid, 1
  %e1 = mul i32 %e0, 5
  br label %merge

merge:
  %v = phi i32 [ %t1, %then ], [ %e1, %else ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

## 9.3 After

```llvm name=region-ifconvert-after.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @region_ifcvt_after(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp slt i32 %tid, 16

  %t0 = add i32 %tid, 1
  %t1 = mul i32 %t0, 3

  %e0 = sub i32 %tid, 1
  %e1 = mul i32 %e0, 5

  %v = select i1 %cond, i32 %t1, i32 %e1
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

This removes branch divergence but executes both sides.

Profitability model:

```text name=ifconversion-profitability.txt
Benefit:
  saved divergent branch/reconvergence cost
  saved mask stack manipulation
  improved scheduling
  simplified CFG

Cost:
  execute both paths
  increased instruction count dynamically
  increased register pressure
  possible lower occupancy
  possible memory speculation hazards

Good candidates:
  small arithmetic-only diamonds
  low-latency instructions
  PHI-heavy merge blocks
  high probability of lane disagreement

Bad candidates:
  large branches
  memory-heavy branches
  calls
  atomics
  barriers
  convergent operations
  branches with high uniformity at runtime
```
# 10. Uniform Branch Extraction

When a divergent region contains a uniform branch inside, it can be beneficial to extract or preserve that branch as scalar control.

Before:

```llvm name=uniform-inside-divergent-before.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @uniform_inside_div(ptr addrspace(1) %out, i32 %mode) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %dcond = icmp ult i32 %tid, 32
  br i1 %dcond, label %active, label %inactive

active:
  %ucond = icmp eq i32 %mode, 1
  br i1 %ucond, label %mode1, label %mode0

mode1:
  %a = add i32 %tid, 100
  br label %join_active

mode0:
  %b = add i32 %tid, 200
  br label %join_active

join_active:
  %v = phi i32 [ %a, %mode1 ], [ %b, %mode0 ]
  store i32 %v, ptr addrspace(1) %out
  br label %merge

inactive:
  br label %merge

merge:
  ret void
}
```

`%dcond` is divergent, but `%ucond` is uniform.

A naive divergent-region if-converter might flatten both `%mode1` and `%mode0`, causing both uniform modes to execute.

Better strategy:

```text name=uniform-branch-extraction.txt
Within a divergent region:
  preserve nested uniform branches when they guard expensive code;
  only predicate the divergent component.
```

A possible transformed structure:

```llvm name=uniform-inside-divergent-after.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @uniform_inside_div_after(ptr addrspace(1) %out, i32 %mode) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %dcond = icmp ult i32 %tid, 32
  %ucond = icmp eq i32 %mode, 1
  br i1 %ucond, label %mode1_all, label %mode0_all

mode1_all:
  br i1 %dcond, label %mode1_active, label %merge

mode1_active:
  %a = add i32 %tid, 100
  store i32 %a, ptr addrspace(1) %out
  br label %merge

mode0_all:
  br i1 %dcond, label %mode0_active, label %merge

mode0_active:
  %b = add i32 %tid, 200
  store i32 %b, ptr addrspace(1) %out
  br label %merge

merge:
  ret void
}
```

This duplicates the divergent guard but preserves uniform mode selection.

Profitability depends on:

- size of duplicated divergent condition,
- size of uniform-controlled regions,
- likelihood of mode values,
- target branch and reconvergence cost.
# 11. Loop Transformations Using Uniformity

Loops are more subtle because divergence can occur in:

- loop condition,
- loop body branches,
- loop exits,
- induction variables,
- PHI nodes,
- values carried across iterations.

## 11.1 Uniform loop

```llvm name=uniform-loop.ll
define amdgpu_kernel void @uniform_loop(ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  store i32 %i.next, ptr addrspace(1) %out
  ret void
}
```

If `%n` is uniform:

- `%i` is uniform.
- `%cond` is uniform.
- All lanes execute same number of iterations.

This is favorable for:

- loop-invariant code motion,
- unrolling,
- scalarization,
- uniform memory access optimization,
- avoiding mask stack loops.

## 11.2 Divergent loop exit

```llvm name=divergent-loop-exit.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @div_loop(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %continue ]
  %i.next = add i32 %i, 1
  %cond = icmp ult i32 %i.next, %tid
  br i1 %cond, label %continue, label %exit

continue:
  br label %loop

exit:
  store i32 %i.next, ptr addrspace(1) %out
  ret void
}
```

Here `%tid` is divergent, so loop exit is divergent.

Different lanes leave the loop at different iterations.

This causes **temporal divergence**:

- A value may be uniform at a definition in one iteration structure.
- But when observed outside the loop, lanes may have exited at different times.
- Thus the outside use may be divergent.

LLVM’s generic uniformity implementation tracks this with a temporal divergence list and cycle-exit divergence logic.

## 11.3 Transform: isolate uniform outer loops from divergent inner guards

Suppose a loop trip count is uniform, but work inside is lane-conditional.

```llvm name=uniform-loop-divergent-body.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @uniform_loop_div_body(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %p = icmp eq i32 (and i32 %tid, 1), 0
  br i1 %p, label %then, label %latch

then:
  %v = add i32 %i, %tid
  store i32 %v, ptr addrspace(1) %out
  br label %latch

latch:
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
```

The loop itself is uniform if `%n` is uniform, but `%p` is divergent.

Good transformation choices:

- Preserve the uniform loop structure.
- If-convert only the small divergent body if safe.
- Do not convert the whole loop into a divergent loop.

Potential predicated form:

```llvm name=uniform-loop-divergent-body-after.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @uniform_loop_div_body_after(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = icmp eq i32 (and i32 %tid, 1), 0
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %v = add i32 %i, %tid

  ; A real GPU backend may lower this to a predicated/masked store.
  ; In portable LLVM IR, the store still needs control flow unless masked store
  ; or target-specific predication is available.
  call void @pseudo.masked.store.i32(i1 %p, i32 %v, ptr addrspace(1) %out)

  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

declare void @pseudo.masked.store.i32(i1, i32, ptr addrspace(1))
```

LLVM IR does not have arbitrary predicated scalar stores, so this example uses a pseudo intrinsic. Actual lowering is target-specific.
# 12. Divergent Exits and Temporal Divergence

Temporal divergence is one of the most important advanced concepts.

Consider:

```llvm name=temporal-divergence.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @temporal(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %exit.cond = icmp eq i32 %next, %tid
  br i1 %exit.cond, label %exit, label %loop

exit:
  store i32 %next, ptr addrspace(1) %out
  ret void
}
```

Even if `%i` starts uniform, `%next` at `exit` is divergent because each lane exits at a different iteration.

A transformation that assumes `%next` is uniform outside the loop would be wrong.

LLVM’s generic uniformity implementation has logic for:

- divergent cycle exits,
- cycles assumed divergent,
- temporal divergence propagation,
- uses outside cycles.

Key mental model:

```text name=temporal-divergence-model.txt
Inside a loop:
  a definition can appear syntactically uniform.

Across a divergent exit:
  different lanes can observe definitions from different dynamic iterations.

Therefore:
  the outside use is divergent even if the static expression looks uniform.
```

This affects:

- LICM,
- loop unswitching,
- exit value simplification,
- scalarization,
- code sinking/hoisting,
- branch folding,
- select conversion.
# 13. Irreducible Control Flow

GPU compilers strongly prefer reducible control flow because reconvergence is easier to model.

Irreducible CFGs can arise from:

- gotos,
- switch lowering,
- aggressive CFG transformations,
- unstructured source languages,
- optimization side effects.

Uniformity analysis in LLVM uses `CycleInfo`, not just classic `LoopInfo`, because it must reason about reducible and irreducible cycles.

For divergent branches in irreducible cycles, the analysis may conservatively mark entire cycles divergent.

Transformation guidance:

```text name=irreducible-guidance.txt
If a CFG region is irreducible:
  1. Avoid delicate uniformity-preserving transformations inside it.
  2. Prefer converting irreducible control flow into reducible loops first.
  3. Recompute UniformityInfo after CFG changes.
  4. Be conservative with hoisting/sinking across cycle boundaries.
```

LLVM’s AMDGPU pipeline includes irreducible-control-flow cleanup such as `FixIrreducible`.
# 14. Convergent Operations and Control Flow

Many GPU operations are **convergent**:

- barriers,
- subgroup collectives,
- warp/wave intrinsics,
- reductions,
- shuffles,
- ballots,
- some target-specific intrinsics.

LLVM has dedicated documentation on convergent operations:

- [`llvm/docs/ConvergentOperations.rst`](https://github.com/llvm/llvm-project/blob/5249e5527f48e1dd6b13c3dabed4992b299bfcb9/llvm/docs/ConvergentOperations.rst)

A control-flow transformation must not illegally:

- duplicate convergent operations,
- move convergent operations across control-flow boundaries,
- speculate convergent operations,
- change which dynamic set of lanes reaches the operation together.

Bad transformation:

```llvm name=bad-convergent-transform-before.ll
declare void @llvm.amdgcn.s.barrier() convergent
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @bad_barrier(i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, %n
  br i1 %cond, label %then, label %merge

then:
  call void @llvm.amdgcn.s.barrier()
  br label %merge

merge:
  ret void
}
```

You cannot simply speculate the barrier:

```llvm name=bad-convergent-transform-after.ll
declare void @llvm.amdgcn.s.barrier() convergent
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @bad_barrier_after(i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, %n

  ; INVALID idea: executing barrier unconditionally changes convergence behavior.
  call void @llvm.amdgcn.s.barrier()

  ret void
}
```

Rule:

```text name=convergent-rule.txt
Never speculate, duplicate, sink, hoist, or if-convert a region containing
convergent operations unless the transformation is explicitly proven legal
under LLVM convergence semantics.
```
# 15. A Concrete Pass Design: Divergence-Aware CFG Optimization

Let us design a hypothetical LLVM function pass:

```text name=pass-name.txt
DivergenceAwareControlFlowPass
```

Goal:

- Use `UniformityInfo` to classify branches.
- Preserve/optimize uniform branches as scalar control.
- If-convert small divergent diamonds.
- Factor mixed uniform/divergent conditions.
- Hoist uniform computations out of divergent regions.
- Avoid unsafe transformations around convergent operations.

## 15.1 Required analyses

For LLVM IR:

```cpp name=required-analyses.cpp
DominatorTreeAnalysis
PostDominatorTreeAnalysis          // useful for regions
LoopAnalysis / CycleAnalysis
TargetIRAnalysis                   // gives TTI
UniformityInfoAnalysis
AssumptionAnalysis                 // optional
AAResultsAnalysis                  // optional for memory safety
MemorySSAAnalysis                  // optional
BlockFrequencyAnalysis             // optional profitability
BranchProbabilityAnalysis          // optional profitability
```

Uniformity itself depends on:

- `TargetTransformInfo`
- `DominatorTree`
- `CycleAnalysis`

## 15.2 High-level algorithm

```text name=divergence-aware-cfg-algorithm.txt
For each function F:
  TTI = TargetIRAnalysis(F)

  if !TTI.hasBranchDivergence(F):
    return no-op or ordinary CPU CFG optimization

  UI = UniformityInfoAnalysis(F)
  DT = DominatorTree(F)
  PDT = PostDominatorTree(F)
  CI = CycleInfo(F)

  For each conditional branch Br:
    Cond = Br.condition

    if UI.isUniform(Cond):
      handleUniformBranch(Br)
    else:
      handleDivergentBranch(Br)

  Recompute analyses after CFG mutation.
```

Important:

> LLVM’s generic uniformity implementation notes that transforms generally should not preserve uniformity after CFG changes. Recompute it.

## 15.3 Uniform branch handling

```text name=handle-uniform-branch.txt
handleUniformBranch(Br):
  1. Avoid if-conversion unless very cheap.
  2. Try to sink divergent computations into the taken regions.
  3. Try to hoist common uniform computations above the branch.
  4. Try uniform loop unswitching if branch is loop-invariant and uniform.
  5. Prefer scalar memory paths if target supports scalar loads.
```

## 15.4 Divergent branch handling

```text name=handle-divergent-branch.txt
handleDivergentBranch(Br):
  1. Identify single-entry/single-exit region.
  2. Reject if region contains:
       - convergent calls,
       - barriers,
       - atomics unless predication is legal,
       - volatile operations,
       - exceptions,
       - indirect branches,
       - large loops.
  3. Estimate cost:
       benefit = divergent branch/reconvergence cost
       cost = extra executed instructions + register pressure
  4. If profitable:
       if-convert region using select/predication.
  5. Else:
       preserve structured branch and maybe canonicalize for backend.
```
# 16. Pseudo-C++ Implementation Sketch

```cpp name=DivergenceAwareControlFlowPass.cpp
PreservedAnalyses
DivergenceAwareControlFlowPass::run(Function &F,
                                    FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);

  if (!TTI.hasBranchDivergence(&F))
    return PreservedAnalyses::all();

  auto &UI = AM.getResult<UniformityInfoAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);

  SmallVector<BranchInst *, 16> Branches;

  for (BasicBlock &BB : F) {
    if (auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (BI->isConditional())
        Branches.push_back(BI);
    }
  }

  bool Changed = false;

  for (BranchInst *BI : Branches) {
    Value *Cond = BI->getCondition();

    if (UI.isUniform(Cond)) {
      Changed |= optimizeUniformBranch(*BI, UI, DT, PDT, TTI);
    } else {
      Changed |= optimizeDivergentBranch(*BI, UI, DT, PDT, TTI);
    }

    if (Changed) {
      // In a real pass, either collect transformations and apply carefully,
      // or invalidate/recompute analyses after mutation.
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  // Do not preserve UniformityInfo after CFG mutation.
  return PA;
}
```

Potential helper:

```cpp name=branch-classification.cpp
enum class BranchUniformity {
  Uniform,
  Divergent,
  Unknown
};

BranchUniformity classifyBranch(BranchInst &BI, UniformityInfo &UI) {
  if (!BI.isConditional())
    return BranchUniformity::Uniform;

  Value *Cond = BI.getCondition();

  if (UI.isUniform(Cond))
    return BranchUniformity::Uniform;

  return BranchUniformity::Divergent;
}
```
# 17. Mixed Predicate Factoring Algorithm

Consider boolean trees:

```text name=boolean-factorization.txt
C = U && D
C = U || D
C = (U1 && U2) && D
C = U && (D1 || D2)
```

Where:

- `U` is uniform,
- `D` is divergent.

Goal:

- Pull uniform decisions outward.
- Delay divergent control until necessary.

Algorithm:

```text name=predicate-factorization-algorithm.txt
factorCondition(C):
  recursively inspect AND/OR tree

  if C = A && B:
    if A uniform and B divergent:
      emit:
        br A, eval_B, false_dest
    if B uniform and A divergent:
      emit:
        br B, eval_A, false_dest

  if C = A || B:
    if A uniform and B divergent:
      emit:
        br A, true_dest, eval_B
    if B uniform and A divergent:
      emit:
        br B, true_dest, eval_A

  preserve short-circuit semantics if source-level semantics are represented
  preserve poison/undef/freeze rules in LLVM IR
```

LLVM IR caveat:

- `and i1` and `or i1` are not necessarily short-circuiting.
- Replacing `and` with branches can change poison behavior unless `freeze` or dominance/proof conditions are handled.
- Be careful with `select`, `and`, `or`, `icmp`, and poison.

Safer version:

```text name=poison-safe-factorization.txt
Only factor conditions if:
  - operands are guaranteed not poison, or
  - inserted freeze preserves intended semantics, or
  - original control dependence already imposed equivalent semantics.
```

Example with freeze:

```llvm name=freeze-factorization.ll
define amdgpu_kernel void @factor_freeze(i1 %u, i1 %d) {
entry:
  %u.fr = freeze i1 %u
  br i1 %u.fr, label %check_d, label %false

check_d:
  %d.fr = freeze i1 %d
  br i1 %d.fr, label %true, label %false

true:
  ret void

false:
  ret void
}
```
# 18. Uniform Loop Unswitching for GPU

Loop unswitching moves loop-invariant branches outside loops.

With uniformity analysis, we can distinguish:

- uniform invariant branches: good unswitch candidates,
- divergent invariant branches: risky; can duplicate divergent loops,
- loop-varying divergent branches: generally not unswitchable.

Before:

```llvm name=uniform-unswitch-before.ll
define amdgpu_kernel void @unswitch(ptr addrspace(1) %out, i32 %n, i1 %flag) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %latch ]
  br i1 %flag, label %then, label %else

then:
  %a = add i32 %i, 1
  br label %latch

else:
  %b = add i32 %i, 2
  br label %latch

latch:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
  %next = add i32 %i, 1
  %c = icmp slt i32 %next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
```

If `%flag` is uniform and loop-invariant:

After:

```llvm name=uniform-unswitch-after.ll
define amdgpu_kernel void @unswitch_after(ptr addrspace(1) %out, i32 %n, i1 %flag) {
entry:
  br i1 %flag, label %loop.then, label %loop.else

loop.then:
  %i.t = phi i32 [ 0, %entry ], [ %next.t, %loop.then ]
  %a = add i32 %i.t, 1
  %next.t = add i32 %i.t, 1
  %c.t = icmp slt i32 %next.t, %n
  br i1 %c.t, label %loop.then, label %exit

loop.else:
  %i.e = phi i32 [ 0, %entry ], [ %next.e, %loop.else ]
  %b = add i32 %i.e, 2
  %next.e = add i32 %i.e, 1
  %c.e = icmp slt i32 %next.e, %n
  br i1 %c.e, label %loop.else, label %exit

exit:
  ret void
}
```

This is attractive if `%flag` is uniform because the whole wave chooses one loop version.

If `%flag` is divergent, this can be terrible:

- different lanes enter different loop copies,
- reconvergence becomes harder,
- code size increases,
- temporal divergence may worsen.

Rule:

```text name=gpu-unswitch-rule.txt
Prefer loop unswitching for uniform loop-invariant conditions.
Avoid or heavily restrict unswitching for divergent conditions.
```
# 19. Branch Probability vs Uniformity

Branch probability and uniformity are different.

A branch can be:

| Branch | Probability | Uniformity | Meaning |
|---|---:|---|---|
| likely true | 99% | divergent | most lanes likely true, but some may differ |
| 50/50 | uniform | uniform | whole wave chooses same side, but runtime side unknown |
| 99% | uniform | uniform | whole wave almost always chooses true |
| 50/50 | divergent | divergent | worst case for SIMT branching |

For GPU CFG transforms, uniformity often matters more than classical predictability.

A uniform 50/50 branch is not necessarily bad on GPU because lanes agree.

A divergent 99/1 branch may still be expensive if minority lanes force serial execution of both paths.

Profitability model should include:

```text name=branch-cost-model.txt
cost(branch) =
  isDivergent ? reconvergence_cost + path_serialization_cost : scalar_branch_cost

path_serialization_cost approximates:
  active_lane_fraction_then * cost_then
  + active_lane_fraction_else * cost_else
  + mask_overhead
```

Static uniformity analysis does not know active lane distribution. Profile data can refine this.
# 20. Memory Operations and Divergence

Memory operations have several dimensions:

## 20.1 Uniform address, uniform value

```llvm name=uniform-store-shape.ll
; all lanes store same value to same address
store i32 %uniform_value, ptr addrspace(1) %uniform_ptr
```

This may be:

- redundant across lanes,
- unsafe to collapse unless memory model allows it,
- optimizable for scalar memory on some targets.

## 20.2 Divergent address, uniform value

```llvm name=divergent-address-uniform-value.ll
%ptr = getelementptr i32, ptr addrspace(1) %base, i32 %tid
store i32 %uniform_value, ptr addrspace(1) %ptr
```

This is common: each lane writes same value to different address.

## 20.3 Uniform address, divergent value

```llvm name=uniform-address-divergent-value.ll
store i32 %tid, ptr addrspace(1) %uniform_ptr
```

This is a race unless controlled by masks/atomics/single active lane.

## 20.4 Divergent address, divergent value

```llvm name=divergent-address-divergent-value.ll
%ptr = getelementptr i32, ptr addrspace(1) %base, i32 %tid
store i32 %tid, ptr addrspace(1) %ptr
```

This is typical per-lane memory access.

Control-flow transforms around memory require caution:

```text name=memory-transform-rules.txt
Do not speculate loads unless:
  - dereferenceability is proven,
  - no trap/poison/UB issue,
  - memory ordering is safe,
  - address spaces allow it,
  - target cost model approves.

Do not speculate stores unless:
  - they can be converted to predicated/masked stores,
  - or they are proven safe and equivalent.

Do not duplicate atomics/volatile operations.
```
# 21. Select vs Branch in LLVM IR

LLVM `select` is useful for data predication:

```llvm name=select-example.ll
%v = select i1 %cond, i32 %a, i32 %b
```

But `select` cannot replace arbitrary control flow with side effects.

Valid:

```llvm name=select-valid.ll
%a = add i32 %x, 1
%b = sub i32 %x, 1
%v = select i1 %cond, i32 %a, i32 %b
```

Not directly valid:

```llvm name=select-invalid-store.ll
; Cannot represent this as a plain select:
if (%cond)
  store %a, %p
else
  store %b, %q
```

Possible alternatives:

- keep branch,
- use masked store intrinsic,
- lower later to target predicated store,
- transform only if addresses are same and store value can be selected:

Before:

```llvm name=store-same-address-before.ll
br i1 %cond, label %then, label %else

then:
  store i32 %a, ptr addrspace(1) %p
  br label %merge

else:
  store i32 %b, ptr addrspace(1) %p
  br label %merge

merge:
  ret void
```

After:

```llvm name=store-same-address-after.ll
%v = select i1 %cond, i32 %a, i32 %b
store i32 %v, ptr addrspace(1) %p
ret void
```

This can be good for divergent branches if both stores are otherwise equivalent.
# 22. PHI Translation During If-Conversion

A diamond PHI:

```llvm name=phi-before.ll
merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
```

Becomes:

```llvm name=phi-after.ll
%v = select i1 %cond, i32 %a, i32 %b
```

But be careful with edge direction:

```text name=phi-edge-rule.txt
If branch is:
  br i1 %cond, label %then, label %else

and PHI is:
  %v = phi [A, %then], [B, %else]

then:
  %v = select i1 %cond, A, B
```

For multiple PHIs:

```llvm name=multi-phi-before.ll
merge:
  %x = phi i32 [ %x.t, %then ], [ %x.e, %else ]
  %y = phi float [ %y.t, %then ], [ %y.e, %else ]
```

After:

```llvm name=multi-phi-after.ll
%x = select i1 %cond, i32 %x.t, i32 %x.e
%y = select i1 %cond, float %y.t, float %y.e
```
# 23. Uniformity-Aware SimplifyCFG

A GPU-oriented `SimplifyCFG` policy could be:

```text name=uniformity-aware-simplifycfg.txt
For a conditional branch:
  if condition is uniform:
    - prefer preserving branch,
    - fold empty blocks,
    - merge identical successors,
    - do not aggressively speculate.

  if condition is divergent:
    - consider converting small diamonds to select,
    - merge stores to same address,
    - convert simple returns to unified return,
    - avoid creating irreducible CFG.

For switches:
  if switch condition is uniform:
    - normal jump table / branch tree may be fine.
  if switch condition is divergent:
    - consider lowering to predicated comparisons only for small switches,
    - otherwise preserve structured multiway branch for backend.
```
# 24. Function Exit Unification

Divergent returns are difficult for SIMT.

Example:

```llvm name=divergent-return-before.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @div_return(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, 16
  br i1 %cond, label %ret1, label %ret2

ret1:
  store i32 1, ptr addrspace(1) %out
  ret void

ret2:
  store i32 2, ptr addrspace(1) %out
  ret void
}
```

A common transformation is to unify exits:

```llvm name=divergent-return-after.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @div_return_unified(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, 16
  br i1 %cond, label %ret1, label %ret2

ret1:
  store i32 1, ptr addrspace(1) %out
  br label %exit

ret2:
  store i32 2, ptr addrspace(1) %out
  br label %exit

exit:
  ret void
}
```

This provides a single reconvergence point.

AMDGPU has logic around unifying divergent function exit nodes in the pipeline.
# 25. Interaction with AMDGPU Scalar vs Vector Registers

AMDGPU has an important hardware distinction:

- SGPR: scalar register, one value shared by wave.
- VGPR: vector register, per-lane value.

Uniformity analysis helps decide whether a value can live in SGPR or must live in VGPR.

Control-flow decisions affect this:

```text name=amdgpu-sgpr-vgpr.txt
Uniform value:
  can often be represented in SGPR.

Divergent value:
  generally requires VGPR.

Uniform branch:
  can often be scalar branch.

Divergent branch:
  requires vector condition / exec mask manipulation.
```

A bad transformation can accidentally make uniform values divergent by placing them behind divergent control and merging through PHIs.

Example:

```llvm name=uniform-to-divergent-phi.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @bad_phi(ptr addrspace(1) %out, i32 %u) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp ult i32 %tid, 16
  br i1 %cond, label %then, label %else

then:
  %a = add i32 %u, 1
  br label %merge

else:
  %b = add i32 %u, 1
  br label %merge

merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

Even though both sides compute the same uniform expression, a naive analysis/transform may create a PHI in a divergent join. Better:

```llvm name=uniform-to-divergent-phi-fixed.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @good_phi(ptr addrspace(1) %out, i32 %u) {
entry:
  %v = add i32 %u, 1
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

Rule:

```text name=avoid-fake-divergence-rule.txt
Avoid introducing PHIs for uniform equivalent expressions at divergent joins.
Hoist or CSE uniform expressions before divergent branches when legal.
```
# 26. Practical LLVM Pass Pipeline Placement

Uniformity-aware CFG transformations should run after enough canonicalization and before target lowering decisions that need structured CFG.

A possible placement:

```text name=possible-pipeline.txt
Early IR:
  - mem2reg
  - instcombine
  - simplifycfg
  - loop-simplify
  - lcssa

Mid/Late GPU IR:
  - infer address spaces
  - code sinking
  - cycle analysis
  - uniformity analysis
  - divergence-aware CFG transform
  - structurize CFG / fix irreducible
  - late simplifycfg with GPU-aware policy
  - codegen prepare

Machine IR:
  - machine uniformity analysis
  - branch selection
  - if conversion / predication
  - exec mask lowering
  - scheduling
```

Important:

```text name=analysis-invalidation.txt
Any pass that changes CFG or value definitions should invalidate UniformityInfo.
Recompute UniformityInfo after nontrivial CFG transformation.
```

LLVM’s `GenericUniformityImpl.h` explicitly notes that transforms generally should not preserve uniformity info because there is no general update interface.
# 27. Correctness Checklist

Before applying a uniformity-based control-flow transform, check:

## CFG legality

- Does the transformed CFG preserve dominance?
- Are PHIs updated correctly?
- Are loop headers/latches still valid?
- Is LCSSA preserved if required?
- Are unreachable blocks removed or handled?

## SSA legality

- Do all uses have dominating definitions?
- Are PHIs replaced with selects correctly?
- Are edge values preserved?
- Are poison/freeze semantics preserved?

## Memory legality

- No illegal load speculation.
- No store duplication unless safe.
- No volatile duplication.
- No atomic duplication.
- No memory ordering violation.

## Convergence legality

- No illegal movement of convergent operations.
- No barrier speculation.
- No subgroup collective duplication.
- No change to dynamic convergence set.

## Uniformity legality

- Uniform values remain uniform where intended.
- Divergent exits are respected.
- Temporal divergence is not ignored.
- Irreducible cycles are treated conservatively.

## Profitability

- Divergent branch removed or reduced.
- Register pressure acceptable.
- Code size acceptable.
- Occupancy not harmed.
- Memory coalescing not worsened.
- Target-specific scalar/vector register benefits considered.
# 28. Example: Full Transformation Walkthrough

## 28.1 Input

```llvm name=full-example-before.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @example(ptr addrspace(1) %out, i32 %n, i32 %mode) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()

  %u0 = icmp sgt i32 %n, 0
  %d0 = icmp ult i32 %tid, 32
  %cond = and i1 %u0, %d0

  br i1 %cond, label %then, label %else

then:
  %common.t = mul i32 %mode, 4
  %a = add i32 %common.t, %tid
  br label %merge

else:
  %common.e = mul i32 %mode, 4
  %b = sub i32 %common.e, %tid
  br label %merge

merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

Assume:

- `%tid` is divergent.
- `%n` and `%mode` are uniform.
- `%u0` is uniform.
- `%d0` is divergent.
- `%cond` is divergent.

## 28.2 Analysis

```text name=full-example-analysis.txt
%tid       divergent
%n         uniform
%mode      uniform
%u0        uniform
%d0        divergent
%cond      divergent because it depends on %d0

Branch on %cond is divergent.

%common.t and %common.e are uniform and equivalent.
%a and %b are divergent because they use %tid.
%v is divergent because it is a PHI at a divergent join.
```

## 28.3 Transformation 1: factor uniform condition

```llvm name=full-example-factor-uniform.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @example_factor(ptr addrspace(1) %out, i32 %n, i32 %mode) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %u0 = icmp sgt i32 %n, 0
  br i1 %u0, label %check_div, label %else

check_div:
  %d0 = icmp ult i32 %tid, 32
  br i1 %d0, label %then, label %else

then:
  %common.t = mul i32 %mode, 4
  %a = add i32 %common.t, %tid
  br label %merge

else:
  %common.e = mul i32 %mode, 4
  %b = sub i32 %common.e, %tid
  br label %merge

merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

Benefit:

- If `%u0` is false, the whole wave skips `%d0` and divergent branch.

## 28.4 Transformation 2: hoist common uniform expression

```llvm name=full-example-hoist.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @example_hoist(ptr addrspace(1) %out, i32 %n, i32 %mode) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %common = mul i32 %mode, 4
  %u0 = icmp sgt i32 %n, 0
  br i1 %u0, label %check_div, label %else

check_div:
  %d0 = icmp ult i32 %tid, 32
  br i1 %d0, label %then, label %else

then:
  %a = add i32 %common, %tid
  br label %merge

else:
  %b = sub i32 %common, %tid
  br label %merge

merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ], [ %b, %entry ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

The above IR as written is structurally problematic because `%b` is not defined on the `%entry -> else` path. A correct version needs a separate block or compute `%b` in `else`.

Corrected:

```llvm name=full-example-hoist-correct.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @example_hoist_correct(ptr addrspace(1) %out, i32 %n, i32 %mode) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %common = mul i32 %mode, 4
  %u0 = icmp sgt i32 %n, 0
  br i1 %u0, label %check_div, label %else

check_div:
  %d0 = icmp ult i32 %tid, 32
  br i1 %d0, label %then, label %else

then:
  %a = add i32 %common, %tid
  br label %merge

else:
  %b = sub i32 %common, %tid
  br label %merge

merge:
  %v = phi i32 [ %a, %then ], [ %b, %else ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

## 28.5 Transformation 3: if-convert divergent inner diamond

The divergent branch is now only in `check_div`.

```llvm name=full-example-final.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @example_final(ptr addrspace(1) %out, i32 %n, i32 %mode) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %common = mul i32 %mode, 4
  %u0 = icmp sgt i32 %n, 0
  br i1 %u0, label %active, label %uniform_false

active:
  %d0 = icmp ult i32 %tid, 32
  %a = add i32 %common, %tid
  %b.active = sub i32 %common, %tid
  %v.active = select i1 %d0, i32 %a, i32 %b.active
  br label %merge

uniform_false:
  %b.false = sub i32 %common, %tid
  br label %merge

merge:
  %v = phi i32 [ %v.active, %active ], [ %b.false, %uniform_false ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

This preserves the outer uniform branch and removes the inner divergent branch.
# 29. Common Mistakes

## Mistake 1: Treating all branches as equally bad

Wrong:

```text name=mistake-all-branches-bad.txt
All branches hurt GPU performance, so convert all branches to selects.
```

Correct:

```text name=correct-branch-view.txt
Divergent branches are expensive.
Uniform branches are often cheap and useful.
```

## Mistake 2: Ignoring sync dependence

A PHI can be divergent even if its incoming values are constants.

```llvm name=constant-phi-divergent.ll
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @constant_phi(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %c = icmp ult i32 %tid, 16
  br i1 %c, label %a, label %b

a:
  br label %m

b:
  br label %m

m:
  %v = phi i32 [ 1, %a ], [ 2, %b ]
  store i32 %v, ptr addrspace(1) %out
  ret void
}
```

`%v` is divergent because different lanes choose different constants.

## Mistake 3: Ignoring temporal divergence

Values exiting divergent loops may be divergent even if they look uniform.

## Mistake 4: Speculating convergent operations

Never speculate barriers or subgroup collectives without convergence proof.

## Mistake 5: Preserving stale UniformityInfo after CFG changes

After transformation, recompute.
# 30. Summary Rules

```text name=summary-rules.txt
1. Use TTI.hasBranchDivergence() to decide whether GPU-style analysis matters.

2. Use UniformityInfo:
     - UI.isUniform(V)
     - UI.isDivergent(V)
     - UI.hasDivergentTerminator(B)

3. Preserve uniform branches when they guard expensive work.

4. If-convert small divergent diamonds when safe and profitable.

5. Hoist common uniform expressions out of divergent regions.

6. Sink divergent work behind uniform guards when profitable.

7. Factor mixed conditions:
     U && D -> branch on U first, then D
     U || D -> branch on U first, then D

8. Treat loops carefully:
     - uniform loop exits are good,
     - divergent loop exits cause temporal divergence.

9. Be conservative with irreducible control flow.

10. Never illegally move, duplicate, or speculate convergent operations.

11. Recompute UniformityInfo after CFG changes.

12. Use target-specific cost models:
     - AMDGPU SGPR/VGPR effects,
     - NVPTX warp divergence,
     - memory coalescing,
     - occupancy/register pressure.
```
