# LLVM Cost Models — Types, Where They Live, and Examples

LLVM  has **several cost-modeling** used by different middle-end and back-end passes. The most commonly referenced ones are:
- **Target Transform Info (TTI) cost model** (IR-level “how expensive is this op on this target?”)
- **Loop Vectorizer cost model** (profitability and vector width selection)
- **SLP Vectorizer cost model** (profitability of packing scalar ops into vectors)
- **Inline cost model** (inliner profitability)
- **MachineScheduler / CodeGen scheduling models** (back-end instruction scheduling / itineraries)
- **Register allocation / spill cost heuristics** (cost of keeping values in registers vs spilling)
- **Instruction selection / pattern selection “cost” heuristics** (choose cheaper machine patterns)
- **Analysis-based (BPI/BFI) profile-driven “cost”** (not a direct op-cost model, but used to weigh “hot” vs “cold” decisions)
---

## 1) Target Transform Info (TTI) Cost Model in LLVM — In-Depth Notes (with Examples)

### 1. What is TTI?

**TargetTransformInfo (TTI)** is LLVM’s *target-aware query interface* used by IR (middle-end) optimizations to answer questions like:

- *“How expensive is this IR instruction on this target CPU?”*
- *“Is this vector type legal and efficient?”*
- *“Will this operation lower to a single machine instruction or a long expansion?”*
- *“How costly are shuffles, reductions, gathers, masked ops, and address computations?”*

TTI exists because LLVM’s middle-end optimizations (vectorization, unrolling, inlining heuristics, etc.) should be **target-aware** without embedding CPU-specific logic inside every pass.

### What TTI is **not**
- Not a cycle-accurate simulator.
- Not guaranteed to map 1:1 to real latency/throughput.
- Not universal across passes as “absolute truth”.

TTI’s primary purpose is **relative comparison**:
> If TTI says option A costs less than option B, LLVM assumes A is more profitable *on that target*.

---

### Where TTI fits in the LLVM pipeline

At a high level:

1. **IR optimization pass** (e.g., LoopVectorize, SLPVectorizer, unrolling heuristics) needs profitability estimates.
2. It queries **TTI** for cost/legality.
3. TTI is implemented by each **target backend** (e.g., X86, AArch64, RISC-V), usually using:
   - subtarget feature flags (AVX2, AVX-512, NEON, SVE…)
   - type legality information
   - lowering rules (e.g., “this becomes a libcall”)
   - instruction selection patterns and known costs

TTI forms the “contract” between target-independent IR passes and target-specific knowledge.

---

###  Why LLVM needs a cost model at IR level

Many transformations improve “instruction count” but might hurt real performance due to:

- expensive instructions (division, remainder, transcendental math)
- expensive vector shuffles/permutations
- scalarization when vector types are unsupported
- heavy runtime checks (alias checks, alignment checks)
- memory system realities (loads/stores vs ALU ops)
- target constraints (register pressure, ABI calls, etc.)

TTI helps guide decisions **before** lowering to machine code.

---

###  The “currency” of TTI: cost units

TTI usually returns costs in an **abstract unit** (often “instruction-like” units).

Important properties:
- Costs are designed to be **comparable** within one target and context.
- Cost of 10 isn’t necessarily “10 cycles”.
- Some targets try to approximate throughput/latency, but it’s heuristic.

In practice, passes do comparisons like:
- `Cost(vectorized_loop) + overhead < Cost(scalar_loop)`
- `Cost(pack+vector_op+unpack) < Cost(scalar_ops)`

---

###  What TTI can estimate (categories)

#### 1. Instruction cost (IR operations)
Examples:
- arithmetic: `add`, `mul`, `fadd`, `fmul`, `fdiv`
- comparisons: `icmp`, `fcmp`
- selection: `select`
- casts/conversions: `zext`, `sext`, `trunc`, `bitcast`, `fptosi`, `sitofp`
- vector ops: `shufflevector`, `insertelement`, `extractelement`
- calls: `call` (incl. intrinsics) and potential libcall lowering

#### 2 Memory operation cost
Examples:
- cost of load/store of type `i32`, `i64`, `<4 x float>`, etc.
- sensitivity to alignment
- whether a vector memory operation will become:
  - a single vector load/store (cheap)
  - multiple scalar loads/stores (expensive)
  - gather/scatter sequences (often expensive)

#### 3 Vector legality and preferred widths
TTI can answer:
- Is `<8 x i32>` legal?
- Is `<16 x i8>` legal?
- Will it widen/narrow, split, or scalarize?

#### 4 Special patterns and building blocks
TTI often participates in cost estimation for:
- **reductions** (horizontal add/min/max)
- **gather/scatter**
- **masked loads/stores**
- **interleaved memory ops**
- **address computations** (folding into addressing modes)

---

#### 5 How IR passes typically use TTI (decision pattern)
A simplified profitability framework:
1. Compute scalar cost:
   - sum of per-instruction costs
2. Compute transformed cost:
   - sum of per-instruction costs in new form
3. Add overheads:
   - shuffles, setup, runtime checks, remainder handling
4. Decide:
   - transform if `NewCost + Overhead < OldCost`

TTI supplies the per-instruction and per-pattern costs.

---

#### 6 In-depth examples (with IR-level illustrations)

##### Example 1: Scalar vs vector add (legal vs illegal vector types)

+ Scalar
```llvm
%a = add i32 %x, %y
%b = add i32 %p, %q
%c = add i32 %m, %n
%d = add i32 %u, %v
```

 - Potential vector form
```llvm
%vx = <4 x i32> ...
%vy = <4 x i32> ...
%vr = add <4 x i32> %vx, %vy
```

**TTI considerations:**
- If `<4 x i32>` matches the hardware SIMD width (e.g., 128-bit or 256-bit vectors), TTI returns a low cost for the vector add.
- If the type is not supported, LLVM may “legalize” it by splitting/scalarizing, e.g.:
  - split `<8 x i32>` into two `<4 x i32>` ops
  - or scalarize into 8 scalar adds

**Decision:**
- If vector add cost + packing/unpacking is cheaper than 4 scalar adds → accept.
- Otherwise reject.

---

### Example 2: Division vs strength reduction (shift/magic multiply)

+ IR division
```llvm
%q = sdiv i32 %a, %b
```

Division is typically expensive compared to add/mul.

+ Power-of-two divisor
If `%b = 8`, a transform may replace:
```llvm
%q = ashr i32 %a, 3
```

**TTI reasoning:**
- `Cost(sdiv)` is usually high.
- `Cost(ashr)` is low.
- Therefore, strength reduction is profitable.

+ Constant non-power-of-two divisor
LLVM can transform divide-by-constant into a “magic” multiply+shift sequence:
- Several integer ops but often still cheaper than hardware division on many CPUs.

TTI helps decide if:
- hardware division cost > sequence cost

---

##### Example 3: Shuffle cost can dominate (why “vector != always faster”)

Suppose SLP tries to pack scalars:

```llvm
%v0 = ...
%v1 = ...
%v2 = ...
%v3 = ...
```

Packing into a vector may require:
- `insertelement` chain
- `shufflevector` permutations
- `extractelement` later

Even if the vector ALU op is cheap, shuffles may be expensive.

**TTI is critical here** because shuffle/permutation cost varies drastically by target:
- Some ISAs have cheap lane permutations.
- Others require multiple instructions or have throughput bottlenecks.

**Profitability check (conceptual):**
- Scalar: `4 * Cost(add)`
- Vector: `Cost(pack) + Cost(vector add) + Cost(unpack)`
- Transform only if Vector < Scalar.

---

##### Example 4: Masked operations and predication in vectorization

+ Scalar loop
```c
for (i=0; i<n; i++)
  if (a[i] > 0) b[i] = a[i];
```

Vectorization introduces masks:
- compare -> mask
- masked store (or blend/select + store)

**TTI questions:**
- Does the target have efficient masked load/store?
- Does it lower to a single instruction, or expand into scalar control flow?
- How expensive is `select` and mask handling?

**Outcomes:**
- On a target with strong predication/masked ops → vectorization likely profitable.
- If masked ops expand heavily → vectorization may be rejected.

---

##### Example 5: Gather/scatter vs scalarization (non-contiguous access)

+ Indexed access
```c
b[i] = a[idx[i]];
```

This is not contiguous, so vectorizing may require gather.

**TTI considerations:**
- Is vector gather supported?
- If supported, is it expensive (often yes)?
- If not supported, will it scalarize to:
  - multiple scalar loads
  - inserts into a vector
  - possible extra control flow

**Often:** `Cost(gather)` is high → vectorization not profitable unless loop is very hot or there are additional wins.

---

##### Example 6: Intrinsic lowering: hardware op vs libcall

IR intrinsic:
```llvm
%r = call double @llvm.sqrt.f64(double %x)
```

**TTI may model:**
- If lowered to hardware `sqrt` instruction: moderate cost.
- If lowered to a libcall `sqrt()`:
  - call overhead + ABI constraints + potential pipeline disruption → high cost.

This influences:
- vectorization (vector sqrt availability)
- hoisting (move out of loop if expensive)
- approximation decisions (if allowed by flags)

---

##### Example 7: Addressing mode and “free” operations (folding into memory ops)

Many targets have rich addressing modes. Example conceptually:
- base + index*scale + offset folded into a load/store

IR might compute an address in steps:
```llvm
%idx = mul i64 %i, 4
%ptr = getelementptr i8, ptr %base, i64 %idx
%val = load i32, ptr %ptr
```

**TTI can reflect that some address arithmetic is “effectively free”** if it can be folded into the load/store addressing mode.

This affects transformations like:
- loop strength reduction
- combining GEPs
- choosing induction variable forms

---

#### 8. TTI and vectorizers: how they rely on TTI specifically

+ Loop Vectorizer
Uses TTI (and its own overhead accounting) to decide:
- vectorization factor (VF)
- interleave factor
- whether masked operations are worth it
- whether to scalarize certain instructions within vector loop (partial vectorization)

+ SLP Vectorizer
Uses TTI to estimate:
- tree cost (packed operations)
- shuffle/permute overhead
- whether it is better to keep scalar

**Rule of thumb:** vectorizers are only as good as:
- legality info + shuffle cost + memory cost modeling
which are exactly where TTI provides target-specific knowledge.

---

##### 9. What affects TTI answers (why they differ per CPU)

TTI costs vary based on:
- enabled ISA features (e.g., SSE4.2 vs AVX2 vs AVX-512)
- native vector register width and preferred types
- lowering rules (some IR ops become sequences)
- costs of shuffles/permutes
- memory alignment penalties
- fast-math / strict FP mode constraints

So the same IR can vectorize on one target and not on another.

---

## 10. Practical checklist: interpreting TTI-driven decisions

When you see LLVM choose not to vectorize/unroll or not to form vectors, likely reasons (often reflected through TTI) include:

1. **Vector type illegal** → scalarization cost too high.
2. **Shuffle tax** → too many permutes to pack/unpack.
3. **Masked/gather operations** too expensive.
4. **Expensive instruction** dominates and doesn’t vectorize well.
5. **Runtime checks overhead** outweighs benefit.

---

# X86 TTI Cost Model
---

## 1) What makes X86 TTI special?

Compared to a “generic” target, X86 has:
- Many ISA feature levels (SSE2 → SSE4.2 → AVX → AVX2 → AVX-512 + subfeatures)
- Rich **addressing modes** (base + index*scale + displacement) that can make some address arithmetic “free”
- Expensive or tricky **shuffles/permutations** depending on width and instruction class
- Clear performance cliffs around:
  - integer division/remainder
  - gathers (especially pre-AVX-512)
  - unaligned/misaligned access (usually okay, but can still be costly in some cases)
  - cross-lane shuffles (256-bit AVX2 split into 128-bit lanes for many operations)
- ABI/library-call lowering for some operations (especially strict FP, long double, certain math)

X86 TTI attempts to capture these *at IR level* so that mid-end passes can choose better transforms.

---

## 2) X86 subtarget features drive most TTI answers

TTI costs for the same IR can change drastically with:
- `+sse2`, `+sse4.1`, `+avx`, `+avx2`, `+avx512f`, `+avx512bw`, `+avx512dq`, `+avx512vl`, etc.
- CPU model (e.g., `-mcpu=skylake`, `znver3`, etc.)
- whether vector widths are preferred/penalized (e.g., 256-bit vs 512-bit decisions)
- availability of specific ops:
  - FMA (`vfmadd*`)
  - vector integer multiply (wider sizes)
  - masked operations (AVX-512 masks are a big deal)
  - vector gather/scatter (AVX2 gather exists; AVX-512 improves masking and throughput patterns)

Practical implication:
> **Always interpret “TTI cost” as “TTI cost under the active subtarget features.”**  
A loop might vectorize under `-mattr=+avx2` but not under `+sse2`, and TTI is one reason.

---

## 3) X86 TTI and vector type “legality” (SSE vs AVX vs AVX-512)

### Common “native” vector widths
- SSE: 128-bit XMM registers
- AVX/AVX2: 256-bit YMM registers
- AVX-512: 512-bit ZMM registers

TTI tends to view operations as cheaper when the vector width and element types map neatly to native instructions.

### Example: `<4 x float>` vs `<8 x float>` vs `<16 x float>`

IR:
```llvm
%r = fadd <8 x float> %a, %b
```

- On **SSE-only**: `<8 x float>` typically splits into two `<4 x float>` ops → higher cost.
- On **AVX**: `<8 x float>` maps to one 256-bit `vaddps` → lower cost.
- On **AVX-512**: could use 512-bit for `<16 x float>` as one op, but:
  - some CPUs may have different frequency/throughput behavior for 512-bit ops
  - cost model may penalize 512-bit usage depending on CPU tuning

Vectorizer uses TTI to choose **vectorization factor** and possibly avoid overly wide vectors when not beneficial.

---

## 4) X86 shuffles/permutations: the “shuffle tax” is real

On X86, shuffle cost depends on:
- whether it’s within 128-bit lanes (cheaper on AVX2)
- cross-lane permutations (often more expensive)
- whether it can be done by a single `pshufd`, `pshufb`, `vperm2f128`, `vpermd`, etc.
- whether it requires multiple instructions

### Example: why AVX2 256-bit vectors can be tricky
Many AVX2 operations are effectively two 128-bit lanes glued together. Some permutations across lanes require extra instructions.

**SLP vectorizer** and **Loop vectorizer** rely on TTI’s shuffle cost to decide if:
- packing scalars is profitable
- rearranging data layout is worth it
- vectorizing with a given VF causes too many permutes

**Rule of thumb on X86:**
> If a transformation introduces many `shufflevector` operations, the X86 TTI cost often rises quickly, and LLVM may refuse vectorization.

---

## 5) Addressing modes: “free” arithmetic when folded into loads/stores

X86 has strong addressing modes:  
`[base + index*scale + disp]` where `scale` is 1,2,4,8.

### Example: IR address computation that might be “free”
```llvm
%idx = shl i64 %i, 2          ; i*4
%ptr = getelementptr i8, ptr %base, i64 %idx
%val = load i32, ptr %ptr
```

On X86, the `i*4` can often be folded into the addressing mode of the load:
- the shift/mul may not need a separate instruction in the final assembly

TTI may model this by not charging as much (or sometimes not charging at all) for address arithmetic if it’s likely foldable.

**Why it matters:**
- loop strength reduction decisions
- induction variable forms
- GEP reassociation and address computation hoisting/sinking

---

## 6) Expensive ops on X86: division/remainder and some FP ops

### Integer division and remainder
IR:
```llvm
%q = sdiv i32 %a, %b
%r = srem i32 %a, %b
```

These are typically **high cost** on X86:
- `idiv` is slow and has low throughput

TTI will usually mark them expensive, encouraging:
- strength reduction (shift/magic multiply when divisor is constant)
- hoisting out of loops when possible
- vectorization avoidance if division dominates and doesn’t vectorize well

### Floating-point division and sqrt
`fdiv` and `sqrt` may be more expensive than `fmul`/`fadd`.
Depending on fast-math flags and available instructions, the cost can differ.

---

## 7) Gather/scatter on X86: AVX2 vs AVX-512 reality

### AVX2 gather exists but is often expensive
Indexed load pattern:
```c
b[i] = a[idx[i]];
```

Vectorizer may consider using gather:
- On AVX2: `vgather*` exists but often high latency/low throughput; also mask handling differs.
- On AVX-512: gathers integrate better with masking, and the model may treat them differently (but still not “cheap”).

TTI commonly makes gathers a *high-cost* operation, so LLVM tends to avoid vectorizing such loops unless there’s enough other benefit.

---

## 8) Masking/predication: big jump with AVX-512

### Without AVX-512
Predication often requires:
- vector compares + blends/selects
- sometimes extra moves/shuffles
- masked store may be emulated

### With AVX-512
Many vector operations support mask registers (`k` regs):
- masked loads/stores
- masked arithmetic

TTI often reflects that:
- masked vectorization becomes more attractive under AVX-512
- conditional loops are more likely to vectorize profitably

Example loop:
```c
if (a[i] > 0) b[i] += a[i];
```
Under AVX-512, the cost of mask handling is often lower than on AVX2/SSE, improving vectorization profitability.

---

## 9) Horizontal reductions: dot products / sums and X86-specific costs

Reduction:
```c
sum += a[i];
```

Vectorization produces:
- vector partial sums
- final horizontal add (“reduce”)

On X86, reduction cost depends on:
- available horizontal add instructions (`haddps`) vs shuffle+add sequences
- vector width (128/256/512)
- whether the reduction can be unrolled to hide latency

TTI helps decide:
- VF choice (e.g., 8 floats with AVX vs 4 floats with SSE)
- whether reduction overhead outweighs benefit for small trip counts

---

## 10) Typical “X86 TTI-driven” outcomes you’ll observe

### A) Preferring AVX2 vectors for contiguous loops
For something like:
```c
a[i] = b[i] + c[i];
```
X86 TTI usually says:
- vector loads/stores + `vaddps` are cheap
- vectorization is profitable

### B) Rejecting vectorization when shuffles dominate
If your loop requires frequent permutations (AoS ↔ SoA style conversions), X86 TTI may increase cost enough that LLVM refuses.

### C) Being conservative about gathers
Even if AVX2 has `vgather`, TTI often discourages it unless necessary.

### D) Different decisions between AVX2 and AVX-512
- AVX-512 can make masking and some vector patterns cheaper
- but may also have tuning concerns regarding very wide vectors on some CPUs
So the “best VF” may change.

---

## 11) Practical mini-examples (C → what TTI tends to favor)

### 11.1 Contiguous map loop (good)
```c
for (i=0; i<n; i++)
  y[i] = x[i] * 3.0f + 1.0f;
```
X86 TTI usually favors vectorization because:
- contiguous loads/stores
- simple arithmetic maps to vector instructions
- FMA may reduce cost further if available

### 11.2 Branchy loop without AVX-512 (maybe not)
```c
for (i=0; i<n; i++)
  if (x[i] > 0) y[i] += x[i];
```
On AVX2, mask handling uses blends and may be expensive.
On AVX-512, masked ops are more direct → more likely profitable.

### 11.3 Indexed loads (often not)
```c
for (i=0; i<n; i++)
  y[i] = x[idx[i]];
```
TTI often makes gather/scalarization expensive → vectorization often rejected.

---

---


---

## 2) Loop Vectorizer Cost Model (profitability & vector width selection)

### What it is
The **Loop Vectorizer** has its own cost model (built on top of TTI) to decide:
- whether to vectorize a loop
- what **vectorization factor (VF)** to choose (e.g., 4, 8, 16)
- whether to use runtime checks (alignment, alias checks)
- whether to use interleaving/unrolling alongside vectorization
- whether scalar epilogs / remainder handling is worth it

It estimates cost of:
- scalar loop body
- vectorized body for each candidate VF
- overhead costs:
  - vector loop setup
  - mask handling
  - reductions
  - gathers/scatters (if non-contiguous access)
  - runtime checks

### Example: vectorize a simple SAXPY-like loop
C-like:
```c
for (i=0; i<n; i++)
  a[i] = b[i] + c[i];
```
Vectorizer considers:
- contiguous loads/stores → usually good
- expected speedup roughly ~VF (bounded by memory bandwidth)

It might choose VF=8 on AVX2 (256-bit vectors) for i32 or float, if legal.

### Example: loop with conditional becomes masked vector code
```c
for (i=0; i<n; i++)
  if (x[i] > 0) y[i] += x[i];
```
Vectorization may require:
- vector compare to produce mask
- masked load/store or blend/select
- extra overhead

Cost model decides if masked vector loop beats scalar.

### Example: reduction profitability
```c
sum = 0;
for (i=0; i<n; i++) sum += a[i];
```
Vectorization introduces:
- vector partial sums
- horizontal reduction at end

Cost model includes reduction overhead.

### Key idea
Loop Vectorizer is **a specialized cost model** for loops that uses TTI instruction costs plus loop-specific overhead modeling.

---

## 3) SLP Vectorizer Cost Model (basic-block / statement packing)

### What it is
The **SLP Vectorizer** vectorizes *within a basic block* by:
- finding isomorphic scalar instruction sequences (trees)
- packing them into vector instructions
- inserting shuffles/extracts as needed

Its cost model compares:
- scalar cost of N independent operations
- vector cost (packed op + required shuffles)
- decides if packing is profitable

### Example: pack 4 independent adds
Scalar:
```llvm
%a0 = add i32 %x0, %y0
%a1 = add i32 %x1, %y1
%a2 = add i32 %x2, %y2
%a3 = add i32 %x3, %y3
```
SLP might transform into:
- build vectors `<4 x i32>` from scalars (inserts)
- one vector `add`
- extract results if needed

If inserts/extracts/shuffles are too expensive (or if results remain scalar), cost model may reject.

### Example: shuffles can kill profitability
If operands are not nicely aligned in registers or memory, SLP may need:
- `shufflevector` operations
- extra permutes (costly on some ISAs)

Cost model includes shuffle/permutation cost via TTI.

### Key idea
SLP profitability depends heavily on **shuffle cost** and whether results stay in vector form.

---

## 4) Inline Cost Model (inliner profitability)

### What it is
LLVM’s inliner uses an **inline cost** computed from:
- estimated size/complexity of callee
- callsite properties (attributes, constant args, etc.)
- bonuses for simplifying (constant propagation, removing branches)
- penalties for code growth
- thresholds (default or profile-guided)

This is not an “instruction latency” model; it’s a **code-size / simplification heuristic**.

### Example: alwaysinline vs normal inline heuristics
- Functions marked `alwaysinline` bypass cost model (unless impossible).
- Otherwise, something like:
  - tiny leaf function → inline
  - large function with loops → likely not inline unless hot callsite

### Example: constant argument enables simplification
```c
int f(int x, int flag) {
  if (flag) return x+1;
  else return x-1;
}
```
At callsite `f(a, 1)`, inlining allows constant-folding away the branch, giving a “bonus” in inline cost.

### Key idea
Inliner cost model trades **code growth** vs **optimization opportunity**.

---

## 5) Back-end Scheduling / Machine Model Costs (itineraries / scheduling model)

### What it is
In the back-end, LLVM uses scheduling models (per target) to estimate:
- instruction latency
- reciprocal throughput
- resource usage (ports/pipelines)
- micro-op decomposition (on some targets)

This guides:
- instruction scheduling (MachineScheduler / post-RA scheduler)
- sometimes impacts unrolling decisions at the machine level

### Example: scheduling around a slow `mul` or `div`
If a target’s model says:
- `imul` latency 3 cycles
- `idiv` latency 20+ cycles
Scheduler tries to:
- move independent instructions to cover latency
- avoid resource conflicts

### Key idea
This cost model is more “microarchitectural” than TTI and is used **after lowering** in machine instructions.

---

## 6) Register Allocation Heuristics (spill cost, rematerialization)

### What it is
Register allocation uses heuristics like:
- spill weight / spill cost (how expensive to spill a value to stack and reload)
- live range splitting decisions
- rematerialization (recompute a value instead of spilling)

### Example: spill vs recompute
If a value is:
```llvm
%v = add i32 %a, 1
```
It might be cheaper to recompute `add` later than spill/reload, depending on:
- usage frequency
- loop nesting depth (hotness)
- instruction cost vs memory op cost

This is a “cost model” in the sense of **estimating profitability** for allocation decisions.

---

## 7) Instruction Selection / Pattern “Cost” Heuristics (ISel)
# Instruction Selection Cost Model

costs come from a mix of:
- legality and type-lowering decisions,
- pattern/predicate priority and complexity,
- per-target heuristics about “cheap” instructions and folds,
- and (sometimes) explicit or implicit “cost” comparisons.

**legality** first, then by **cost/profitability heuristics**.
---


## What “cost modelling” means *in ISel context*

In ISel, cost modelling is mostly about choosing among **multiple legal lowerings**.

Typical dimensions:
1. **Instruction count / pattern complexity**
2. **Microarchitectural cost hints** (latency/throughput/resource usage; more prominent post-ISel in scheduling)
3. **Code size** (sometimes prefer shorter encodings)
4. **Folding opportunities** (combine ops into one instruction)
5. **Register pressure / constraints** (special regs, partial regs, flag dependencies)
6. **Addressing modes** (can we fold scales/offsets)
7. **Vector width and shuffle complexity** (permute costs differ a lot)

*prefer patterns that produce fewer / simpler instructions and enable folding.*

---

## Generic cost modelling mechanisms used during ISel

### Legality and type legalization as a “cost signal”
Before comparing two patterns, LLVM must ensure operations are **legal** on the target:
- If an operation/type is illegal, LLVM must **expand** or **legalize** it (split, widen, scalarize, libcall).
- This expansion is effectively a huge cost penalty.

**Example (generic): illegal i128 multiply**
LLVM IR:
```llvm
%p = mul i128 %a, %b
```
If the target has no native i128 multiply:
- ISel/legalizer expands into multiple i64 multiplies and adds (long sequence).
- Even without explicit “cost numbers,” legality drives the outcome:
  - “native single instruction” (if exists) is chosen
  - otherwise “expanded sequence” is the only legal path

**Takeaway:** In ISel, *“legal vs illegal” often dominates cost decisions.*

---

### Pattern selection: “more specific / more complex” often wins
In SelectionDAG ISel, TableGen patterns are matched. Patterns have:
- predicates (`SubtargetFeature` checks)
- complexity ordering and specificity
- sometimes explicit “AddedComplexity”

In GlobalISel, the selector similarly chooses from legal patterns/instruction mappings and prefers more constrained matches.

**Example (generic): fused multiply-add**
IR:
```llvm
%r = fadd float (fmul float %a, %b), %c
```
If target has FMA:
- prefer one `fma` instruction
Else:
- select `mul` then `add`

This is “cost modelling” via:
- fewer instructions,
- better precision semantics (if allowed),
- and a pattern that only exists when features allow.

---

### Combiner phases around ISel (pre-isel and post-isel)
Cost modelling in ISel is also influenced by:
- IR combines before ISel (InstCombine, DAGCombine, MachineCombiner)
- target-specific DAG combines (e.g., fold constants, address patterns)
- post-isel peepholes

These combines often apply rules like:
- *replace expensive op with cheaper sequence*
- *fold ops into addressing modes*
- *choose shorter or faster pattern*

---

### Branch vs select vs conditional move
A classic “cost” decision (sometimes earlier than ISel, but strongly target-shaped):

IR:
```llvm
%r = select i1 %cond, i32 %x, i32 %y
```

Possible lowerings:
- conditional move (`cmov`) style instruction
- branch + phi (control flow)

Generic heuristics consider:
- target support for cmov/predication
- predicted branch probability (if known)
- code size and pipeline effects

Often, the final mapping is done in or near ISel because it’s target-dependent.

---

## X86-specific ISel cost modelling 
X86 is an excellent case study because it offers:
- rich addressing modes
- many vector shuffle instructions
- special flag dependencies (`EFLAGS`)
- multiple instruction forms for the same operation (reg-reg, reg-mem, mem-reg)
- `LEA` as a “free-ish” arithmetic instruction in many contexts
- `CMOV` and `SETcc` patterns
- AVX/AVX2 lane behavior and AVX-512 masking

**choosing the best instruction form** and **maximizing folding**.

---

##  Key examples

### Addressing modes and folding: “free” arithmetic via mem operands

X86 can encode memory operands like:
```
[base + index*scale + disp]
```

#### Example A: fold GEP into a load/store
IR-like intent:
```c
x = *(int*)(base + i*4 + 16);
```

Naive instruction sequence:
- `imul` to scale `i`
- `add` base
- `add` disp
- `load`

X86 ISel often produces:
- a single `mov` (load) using a complex addressing mode:
  - `mov eax, dword ptr [rdi + rsi*4 + 16]`

**Cost model behavior:**
- Folding address computation into the memory op reduces instruction count and register usage.
- X86 ISel strongly prefers patterns that enable such folds.

#### Example B: LEA for arithmetic without affecting flags
If IR needs:
```c
t = a + b*4 + 8;
```
X86 may use:
- `lea rax, [rdi + rsi*4 + 8]`

**Why LEA is “cheap” in X86 modelling:**
- Often single instruction
- Doesn’t clobber flags (unlike `add`/`sub`)
- Great for address-like arithmetic

So ISel frequently prefers LEA-based patterns for add+mul-by-constant combinations.

---

### Reg-reg vs reg-mem forms: fewer instructions vs hidden costs

X86 allows many ops with a memory operand:
- `add eax, [mem]` (load + add in one instruction)

#### Example: add with load folded
IR:
```llvm
%v = load i32, ptr %p
%r = add i32 %v, %x
```

Possible machine forms:
1. separate load then add:
   - `mov eax, [p]`
   - `add eax, x`
2. folded memory operand:
   - `add eax, [p]`

**Cost modelling tradeoff:**
- Folding reduces instruction count and register pressure.
- But it also can:
  - create memory dependencies in ALU ops,
  - reduce scheduling flexibility,
  - sometimes be slower on specific microarchitectures.

LLVM’s X86 backend generally still prefers folds when profitable, but may avoid them in some scenarios (e.g., when it blocks other combines or affects codegen patterns).

---

### Flags (EFLAGS) as a constraint cost

Many X86 integer ops set flags implicitly. Flags are a limited resource:
- using them can create false dependencies
- can restrict scheduling and register allocation

#### Example: compare + branch vs arithmetic setting flags
IR:
```llvm
%cmp = icmp slt i32 %a, %b
br i1 %cmp, label %T, label %F
```

X86 lowering:
- likely uses `cmp` (sets flags) then `jl` (consumes flags)

Now consider if an arithmetic op that sets flags is between cmp and branch:
- it may clobber flags → extra compare needed

**Cost modelling impact:**
- ISel tries to choose instruction sequences that:
  - avoid unnecessary flag clobbers
  - or fuse compare+branch patterns efficiently
- Sometimes prefers `lea` over `add` to avoid flags clobbering if flags are needed soon.

---

### Select lowering: `cmov` vs branch

IR:
```llvm
%r = select i1 %c, i32 %x, i32 %y
```

On X86 with CMOV:
- `cmov` can implement select without a branch.

However, `cmov` is not always best:
- if `%c` is highly predictable, a branch might be cheaper
- `cmov` can force both `%x` and `%y` computations to happen eagerly

**ISel cost modelling:**
- often uses heuristics:
  - size/simplicity
  - whether operands are cheap
  - probability info if available
- final choice can be influenced by later passes too

---

### Multiplication by constant: IMUL vs shifts/adds vs LEA

IR:
```llvm
%r = mul i32 %x, 5
```

Possible lowerings on X86:
- `imul reg, reg, 5`
- `lea reg, [x + x*4]` (x + 4x = 5x)
- `shl` + `add` sequence

**X86 modelling commonly prefers:**
- `lea` for small constant multipliers where it fits the addressing form
- `imul` for general constants or when LEA pattern not possible/beneficial

**Cost factors:**
- instruction count
- latency/throughput (varies by uarch)
- flags clobbering (LEA does not clobber flags)

---

### Vector shuffle selection: SSE/AVX/AVX2/AVX-512 instruction choice

IR vector shuffle:
```llvm
%r = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <...mask...>
```

On X86, there are many shuffle-like instructions:
- `pshufd`, `pshufb`, `unpck*`, `palignr`, `blend*`
- AVX/AVX2: `vperm2f128`, `vpermd`, `vpshufb` (lane restrictions)
- AVX-512: `vpermt2*`, `vpermps`, plus masking

**Cost modelling in ISel:**
- choose the instruction (or sequence) that implements the shuffle mask cheapest
- avoid cross-lane permutations when possible (AVX2 lane split)
- prefer instructions available under the active subtarget features

**Practical outcome:**
- the same IR shuffle mask can lower to:
  - one instruction under AVX-512
  - multiple instructions under AVX2
  - even more under SSE2

That “difference” is a form of cost modelling driven by feature availability and pattern complexity.

---

## 6) How TTI relates to ISel cost modelling (and where the boundary is)

TTI is mainly used by **IR passes** before ISel. But the relationship is important:

- IR transformations guided by TTI (e.g., vectorization, unrolling) create IR patterns.
- ISel then chooses the best X86 instructions for those patterns.

So you can think of a two-level system:

1. **TTI cost model (IR-level)** predicts:
   - “if we create this vector shuffle / gather / masked op, will it be expensive on X86?”
2. **ISel cost heuristics (lowering-level)** decide:
   - “given that we have this shuffle, which exact X86 instruction(s) is cheapest under SSE/AVX/AVX-512?”

### Example: TTI discourages expensive shuffles, ISel still must implement them
- If a transform introduces a complex `shufflevector`,
  - TTI might have been overly optimistic or forced by legality.
- ISel then picks among X86 shuffle instructions (or sequences), but it cannot avoid the fundamental cost.

**Takeaway:** Good TTI modelling reduces the chance that ISel ends up generating shuffle-heavy slow code.

---

## Worked examples: end-to-end view (IR choice → X86 ISel choice)

### Example 1: Scalar address arithmetic vs folded memory operand

#### Source intent
```c
int t = base[i*4 + 4];
```

#### IR-ish
- compute index scale
- gep
- load

#### X86 ISel preference
- fold all into one memory operand:
  - `mov eax, [rdi + rsi*16 + 16]` (example form)

**Cost logic:**
- fewer instructions
- fewer temporary registers
- better code size

---

### Example 2: `select` lowering

#### IR
```llvm
%r = select i1 %c, i32 %x, i32 %y
```

#### X86 choices
- `cmov` sequence (branchless)
- branch + phi (control flow)

**Cost logic:**
- `cmov`: good when branch unpredictable / operands already computed
- branch: good when condition predictable and one side cheap to skip

---

### Example 3: Multiply by 3

#### IR
```llvm
%r = mul i64 %x, 3
```

#### X86 choices
- `lea rax, [rdi + rdi*2]`  (x + 2x)
- `imul rax, rdi, 3`

**Cost logic:**
- LEA often preferred:
  - single instruction
  - no flags
  - good throughput on many CPUs

---

### Example 4: Vector permute availability changes everything

#### IR
A complex permute on `<16 x i32>`

- Under AVX2:
  - may require multiple instructions and lane-crossing ops
- Under AVX-512:
  - might be a single permute with masks

**Cost logic:**
- ISel picks the cheapest legal instruction(s) available.
- Earlier, TTI would also have priced that permute differently and could influence whether the transform happens at all.


---

## 8) Profile/Hotness-weighted “Cost”: BPI/BFI and PGO - Cost Model

+ notion of **cost × hotness** rather than cost alone. Even if an optimization is “profitable” in isolation, LLVM may avoid it in cold code to reduce code size, compile time, or instruction cache pressure; conversely, it may apply more aggressive transforms in hot regions.


## Big picture: “Hotness-weighted cost” is not one pass, but a cross-cutting idea

**Hotness-weighted cost** means:
> Apply these models *more aggressively* (or accept higher code growth) in **hot** regions, and be conservative in **cold** ones.

This concept shows up in:
- Inlining thresholds (hot callsite → inline more)
- Loop unroll/vectorize thresholds (hot loop → do more)
- Code layout and function ordering (hot blocks together)
- If-conversion decisions
- Machine block placement (fallthrough and branch layout)
- Partial inlining and outlining decisions
- Simplifying CFG and speculation decisions

---

## Terminology

### BPI — BranchProbabilityInfo
**BPI** estimates the probability that a branch goes to each successor.

Sources:
- static heuristics (loop backedges likely taken, etc.)
- metadata / profile data (from PGO)

Example:
- Backedge in a loop might be predicted ~90–99% taken (heuristic)
- With PGO it can be 99.99% for big loops or 50% for small loops, etc.

###  BFI — BlockFrequencyInfo
**BFI** estimates the **relative execution frequency** of each basic block in a function.

Key:
- BFI is derived from BPI + CFG structure (and optionally actual counts from PGO)
- BFI gives *relative weights* like “block A runs 100x more than block B”

###  Hot / Cold / Warm classification
LLVM often labels blocks or callsites as:
- **hot**: very frequently executed
- **cold**: rarely executed
- **warm**: in-between (sometimes used)

This classification influences thresholds and heuristics.

---

## 3) Where hotness comes from

### Static heuristics (no profile)
LLVM can still estimate BPI/BFI based on:
- loop structure (backedges likely)
- branch prediction heuristics (e.g., error paths are cold, null checks often not taken, etc.)
- attributes like `cold`, `unlikely`, `__builtin_expect`

This is useful, but not as accurate as real profiles.

###  Instrumentation-based PGO (InstrPGO)
Flow:
1. Compile with instrumentation (`-fprofile-generate` / `-fprofile-instr-generate`)
2. Run representative workloads → generates `.profraw`
3. Merge to `.profdata`
4. Recompile with `-fprofile-use` / `-fprofile-instr-use`

What it gives LLVM:
- real edge counts (branch taken counts)
- real block execution counts
- callsite hotness and call counts
- value profile info for some optimizations (indirect calls, mem ops, etc.)

### Sample-based PGO (SamplePGO)
Flow:
1. Build binary
2. Run under a sampling profiler (e.g., `perf`) → collect samples
3. Convert to profile format (e.g., `llvm-profdata` for samples)
4. Compile with `-fprofile-sample-use`

What it gives LLVM:
- sampled instruction/address hotness
- approximate call graph hotness
- tends to be less precise for edge counts than instrumentation, but can be very effective and cheaper to deploy

---

## The key idea: cost is multiplied or gated by hotness

LLVM does not literally do “cost × frequency” everywhere, but conceptually many decisions behave like:

- Optimize if:
  - `SpeedupBenefit(block) * Hotness(block) > CodeGrowthPenalty`
- Or increase threshold if block/callsite is hot:
  - `InlineThreshold = Base + Bonus(hotness)`
- Or avoid changes in cold blocks even if locally profitable:
  - “don’t vectorize cold loops”
  - “don’t unroll cold loops”
  - “don’t speculate expensive operations in cold paths”


---

## 5) Examples using BPI/BFI (generic)

### Example 1: Cold error handling block
C-like:
```c
int parse(...) {
  if (unlikely(err)) return -1;  // error path
  // hot parsing loop
  ...
}
```

CFG:
- entry → if(err) → error_return
- entry → else → hot_path

**With BPI/BFI:**
- `error_return` block frequency is very low
- `hot_path` is high

Effects:
- keep error handling code small (avoid unrolling, avoid heavy inlining into it)
- prioritize optimizing the hot path

### Example 2: Loop backedge probability drives BFI
```c
for (i=0; i<n; i++) body();
```

Even without PGO:
- backedge likely taken many times
- blocks inside loop are “hotter” than exit blocks

With PGO:
- LLVM may learn actual average `n` and branch behavior, refining BFI and thus the aggressiveness of unrolling/vectorization.

---

## PGO-driven decisions (generic, but widely visible)

### Inlining with hot callsites
Inlining uses an inline cost model (size vs benefit). Hotness changes thresholds.

Example:
```c
int hot_loop(...) {
  for (...) total += f(x); // callsite #1, very hot
}

int cold_path(...) {
  if (err) return f(x);    // callsite #2, cold
}
```

With PGO:
- callsite #1 gets a hotness bonus → inline `f` even if it’s moderately large
- callsite #2 might not inline to avoid bloating cold code

### Code layout (basic block placement / function ordering)
With profile:
- place frequently executed blocks adjacent to reduce taken branches and I-cache misses
- outline cold blocks (or place them out-of-line)
- reorder functions so hot functions are close in memory

This tends to improve instruction cache locality and branch prediction behavior.

### If-conversion / select vs branch
Transform:
- convert branchy code to branchless `select` (or CMOV at machine level)

Hotness decides:
- If branch is unpredictable in a hot region, branchless may win.
- If branch is predictable or cold, keep branch to reduce unnecessary work.

---

##  How “hotness-weighted cost” interacts with X86

In X86 hotness weighting is very impactful:
### Branch prediction and front-end costs
On X86, especially modern OoO cores:
- mispredicted branches are expensive (pipeline flush)
- taken branches can also have front-end cost
- but predictable branches are often very cheap

**Hotness-weighted implication:**
- In *hot* code, LLVM is more willing to restructure control flow (layout, if-conversion) to reduce mispredicts.
- In *cold* code, it may prefer smaller code size even if branchless would be marginally faster.

### Inlining and i-cache pressure on X86
Inlining increases code size, which can hurt I-cache / uop cache behavior.

With PGO:
- inline aggressively into hot loops/functions
- avoid inlining into cold paths to preserve cache locality

X86-specific note:
- Code layout and i-cache behavior are often first-order effects for performance on X86 servers and desktops.

### Vectorization/unrolling tradeoffs
Vectorization/unrolling can:
- increase code size
- increase register pressure
- increase decode bandwidth pressure (more instructions, more bytes)

Hotness tells LLVM where it’s worth paying these costs.

On X86:
- The benefit of vectorization can be large (SSE/AVX/AVX2/AVX-512)
- But code size and front-end throughput can limit real speedups

Thus, profile-guided hotness helps select:
- which loops to vectorize
- how much to unroll
- whether to version loops with runtime checks

### CMOV vs branch (X86 classic)
At machine level, X86 has `cmovcc`.

Hotness-weighted guidance:
- If the condition is unpredictable and code is hot → CMOV/branchless may help
- If condition is highly biased → branch is good
- If code is cold → keep simplest/smallest

PGO provides the *bias* and *frequency* information needed.

---

## Worked example: hot/cold split changes optimization choices 

### Source
```c
int foo(int *a, int n, int flag) {
  int sum = 0;
  for (int i=0; i<n; i++) {
    if (flag) sum += a[i];     // path A
    else      sum -= a[i];     // path B
  }
  return sum;
}
```

#### Without profile
LLVM may guess `flag` is unpredictable:
- consider if-conversion / select-like forms
- might create branchless form inside loop to avoid mispredicts

#### With PGO
Suppose real workload: `flag` is almost always 1.
- Branch becomes highly predictable
- In hot loop, a predictable branch can outperform a branchless select (which does extra work)
- LLVM may keep the branch, or even specialize (if it can) through cloning/versioning

On X86, this can be significant because:
- predictable branches are cheap
- unnecessary arithmetic in hot loops costs throughput

This is “hotness-weighted cost” in action:
- profile says both the *loop is hot* and the *branch is biased*
- so the model favors the predictable branch.

---

## Worked example: inline only in hot path

### Source
```c
static inline int g(int x) { return x * 3 + 1; } // small

int f(int *p, int n) {
  int t = 0;
  for (int i=0;i<n;i++) t += g(p[i]);  // hot
  if (n == 0) return g(7);             // cold-ish
  return t;
}
```

With PGO:
- hot loop callsite: inline `g` (benefit high, overhead removed in hot code)
- cold callsite: even if it inlines, it’s less important; LLVM may still inline because it’s tiny, but for larger `g`, it would avoid cold growth.

X86 angle:
- hot loop benefits from better scheduling and vectorization opportunities when `g` is inlined
- cold site doesn’t justify code growth (i-cache efficiency)

---


---
