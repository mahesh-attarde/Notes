
## 1. What is Global Value Numbering?
**Global Value Numbering (GVN)** is a compiler optimization technique that finds **equivalent computations** in a program and eliminates redundancy.

Informally:  
> If two computations *always* produce the same result at runtime, GVN tries to compute that value **once** and reuse it.

Examples of redundant computations:

```c
int x = a + b;
...
int y = a + b;  // recomputing the same expression
```

If `a` and `b` haven’t changed between those two lines and control-flow conditions guarantee both are executed, `y` can reuse the result of `x`.

In SSA (Static Single Assignment) form, a lot of this is already visible due to unique names, but **semantic equivalence** can go beyond syntactic identity. GVN extends this to more general cases.


## 2. What problem does GVN solve and why?

### Why it exists when we already have SSA and local CSE?

- SSA makes some equivalences obvious: reusing the same SSA variable is trivial.
- But compilers generate SSA from higher-level code; during transformations, many semantically identical values might appear with different SSA names or in different basic blocks.
- Local CSE only sees within a block; it doesn’t reason globally using **dominance** and **control flow**.
- GVN uses program structure and semantics to prove that different expressions are **congruent** (equivalent) and unify them.

## 3. Key Concepts

### 3.1 Value Numbers

A **value number** is an abstract identifier (like an integer) that represents an **equivalence class of expressions** that always produce the same value at runtime.

- If two instructions get the same value number, the algorithm believes they are equivalent.
- A value number is *not* the same as an SSA name; many SSA values can map to the same value number.

Example:

```c
int x = a + b;       // gets value number VN1
int y = a + b;       // also gets VN1 (same expression, same operands)
int z = x;           // z is just a copy; also VN1
```

### 3.2 Expressions and Congruence Classes

An “expression” for GVN is something like:

- `add(VN_of_operand1, VN_of_operand2, flags)`  
- `mul(VN_of_operand3, VN_of_operand4)`  
- `load(memory_state, address_VN)` (a bit more complex for real GVN)

GVN maintains a mapping:

- expression → value number  
- and inversely, value number → set of IR instructions that have this number (congruence class).

**Congruence class** = all instructions that are believed to be semantically equivalent.

### 3.3 Memory and Side Effects

For pure arithmetic ops, equivalence is straightforward. For memory operations:

- Two loads from the same address aren’t necessarily equivalent if something could have stored to that address in between.
- GVN needs some notion of **aliasing** and **memory state** to know whether memory is unchanged.

LLVM uses alias analysis and some memory modeling to decide when loads/stores are safe to treat as equivalent.

## 4. What does LLVM’s GVN pass *actually do*?

LLVM has a `GVN` pass that performs:

1. **Global common subexpression elimination**:
   - If it finds an instruction whose expression has already been computed and dominates it, it replaces it with the dominating value.
   - Example in LLVM IR:

   ```llvm
   %1 = add i32 %a, %b
   ...
   %2 = add i32 %a, %b  ; later, dominated by the first
   ```

   GVN replaces all uses of `%2` with `%1`, and `%2` becomes dead.

2. **Redundant load elimination**:
   - If load from `ptr` is equivalent to an earlier load from `ptr` with no interfering store, reuse the earlier result.

3. **Partial redundancy elimination (PRE) in some versions**:
   - In certain configurations/variants, GVN is augmented to perform **GVN-PRE**, which also *inserts* computations on some paths to remove redundancy overall. (LLVM has a separate `GVNHoist` / `GVNHoistPass` and other related optimizations historically.)
   - Basic GVN is mainly about *replicating* an existing dominating computation, not inserting new ones.

4. **Constant folding and simplification** help indirectly**:
   - LLVM has InstCombine and ConstantFold, but GVN can sometimes trigger simplifications when congruent expressions are replaced.

## 5. How does GVN work algorithmically?

### 5.1 Core High-Level Algorithm

Think in terms of these steps (simplified):

1. **Compute dominance information**:
   - For each basic block, know which blocks dominate it.
   - A value computed in a dominator block is available in the dominated block (if we can prove safety).

2. **Process blocks in a dominance order** (often pre-order):
   - Maintain a **hash table** (or map) from *expressions* to *value numbers/representative instructions*.
   - Maintain a mapping from each IR `Value*` to its value number.

3. **For each instruction** in that order:
   - If it’s something we can handle (pure op, load without side effects, etc.):
     1. Build an **expression key**: type + opcode + operands’ value numbers + relevant flags (e.g., `nsw`, `nuw` for add, etc.).
     2. Look up this expression in the table:
        - If **found**: there is already a dominating equivalent instruction; reuse that instruction’s value:
          - Replace uses of the current instruction with the representative.
          - Mark current instruction as redundant (to be removed).
        - If **not found**: assign a new value number, and insert `(expression → VN)` and `(instruction → VN)`.

   - For memory operations (loads), expression includes:
     - The base pointer value number
     - Some memory state/alias information
     - Maybe alignment / type info

4. **After processing the function**:
   - Remove all instructions that are now unused (DCE).

### 5.2 More Specific Notes for LLVM GVN

LLVM’s GVN is more complex because:

- It handles:
  - `phi` nodes
  - `select` instructions
  - Some forms of `icmp`/`br` conditions
  - Memory dependencies (loads, stores, calls, etc.)
- It uses:
  - **DominatorTree**
  - **MemoryDependenceAnalysis** or **MemorySSA** (historically)
  - **AliasAnalysis (AA)**

Conceptually, LLVM’s GVN:

1. **Builds a worklist of instructions** to consider.
2. For each:
   - Checks if it is “value-numberable” (pure enough / analyzable).
   - Calculates a **hash** representing its expression (opcode + operands + type + flags).
   - Uses a map `Expression -> Instruction*` to detect duplicates.
   - For loads:
     - Uses memory dependency queries to ensure that no clobbering store exists between the previous equivalent load and this one.

3. **Performs simple PRE** in some flavors:
   - If an expression is partially redundant (computed on some but not all incoming paths), it may insert computations on missing paths to make it fully redundant. (This is more advanced and optional; basic understanding can ignore this initially.)


## 6. How GVN interacts with SSA, CFG, and dominance

### 6.1 SSA and GVN

- SSA already ensures each definition is unique; uses are easy to track.
- GVN uses the SSA graph to propagate equivalence:
  - If `%x` and `%y` are equivalent and you compute `add %x, 1` vs `add %y, 1`, then those adds are equivalent.
- **Phi nodes**:
  - `phi` merges values from multiple incoming edges.
  - Two phis may be equivalent if:
    - They have the same structure and
    - Each corresponding incoming value is equivalent.
  - This is trickier and handled carefully in LLVM’s implementation.

### 6.2 Control Flow and Dominance

- GVN is *global* because it sees all basic blocks and uses dominance to know where it can safely reuse previous computations.
- Example:

```c
if (cond) {
    x = a + b;   // BB1
}
...
y = a + b;       // BB2
```

- If BB1 doesn’t dominate BB2, we can’t always reuse `x` directly across all paths (maybe `cond` is false).
- Pure CSE within a block is easy; GVN extends this across the CFG but still must respect control flow.

### 6.3 Memory and Aliasing

For loads:

```c
int *p = ...;
int a = *p;
...
int b = *p;
```

To reuse `a` for `b`, GVN needs to know:

- No store `*p = ...` or other potential alias store happened between the two loads.
- This is where **alias analysis** and **memory dependence analysis** come in:

  - If `store X, q` where `q` may alias `p`, we must be conservative and assume memory changed (unless we can prove otherwise).
  - LLVM’s GVN queries these analyses to ensure the load is *safe* to eliminate.


---
Focusing from `GVNPass::runImpl` and following the actual execution path is exactly the right way to understand “what GVN does, why, and how”.

I’ll walk through the control flow and the main conceptual stages, not line-by-line, and I’ll continually relate the mechanics back to the *idea* of GVN: deduplicate computations and loads by assigning them equivalence classes (“value numbers”) and then CSE / PRE them.

All code references below are to:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2803-L2873
bool GVNPass::runImpl(Function &F, AssumptionCache &RunAC, DominatorTree &RunDT,
                      const TargetLibraryInfo &RunTLI, AAResults &RunAA,
                      MemoryDependenceResults *RunMD, LoopInfo &LI,
                      OptimizationRemarkEmitter *RunORE, MemorySSA *MSSA) {
  ...
}
```

---

## 1. GVN Working
- Maintains a mapping `Value* -> value number` (the *ValueTable* `VN`).
- Maintains, for each value number, a set of “leader” instructions (one or more dominating definitions) in a *LeaderTable*.
- Walks the function top‑down in dominance order, and for each instruction:
  - Tries to locally simplify it;
  - Gives it a value number;
  - Sees if there is an equivalent (same value number) dominating definition; if yes, replaces it (CSE).
  - For loads, uses alias / memdep / MemorySSA to also eliminate redundant loads, including **non‑local** ones via SSA construction (PHI insertion).
  - For conditions and assumes, does some equality propagation (basic constant propagation / predication-like reasoning).
- Repeats this until a fixed point (no more changes).
- Then, optionally, runs a scalar **PRE** step (Partial Redundancy Elimination) built on top of the same value numbering.
- Cleans up and returns whether the function changed.

So GVN is both a pure value-numbering CSE pass *and* a specialized load-PRE + scalar-PRE pass that uses the same infrastructure.

## 2. Entry point: `GVNPass::runImpl`

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2803-L2822
bool GVNPass::runImpl(Function &F, AssumptionCache &RunAC, DominatorTree &RunDT,
                      const TargetLibraryInfo &RunTLI, AAResults &RunAA,
                      MemoryDependenceResults *RunMD, LoopInfo &LI,
                      OptimizationRemarkEmitter *RunORE, MemorySSA *MSSA) {
  AC = &RunAC;
  DT = &RunDT;
  VN.setDomTree(DT);
  TLI = &RunTLI;
  VN.setAliasAnalysis(&RunAA);
  MD = RunMD;
  ImplicitControlFlowTracking ImplicitCFT;
  ICF = &ImplicitCFT;
  this->LI = &LI;
  VN.setMemDep(MD);
  VN.setMemorySSA(MSSA);
  ORE = RunORE;
  InvalidBlockRPONumbers = true;
  MemorySSAUpdater Updater(MSSA);
  MSSAU = MSSA ? &Updater : nullptr;
```

Here GVN:

- Stores analysis interfaces in member variables (DT, AA, TLI, LoopInfo, etc.).
- Hooks them into `VN` (the value numbering table) and memory analysis (`MD`, `MSSA`).
- Sets up `ICF` for implicit control-flow (guards, exceptions) used to decide if speculation is safe.
- If MemorySSA is available, creates a `MemorySSAUpdater` to keep it consistent when GVN moves/inserts/removes memory accesses.

**Why:** GVN is fundamentally a *global* analysis/transform. It needs dominance, alias info, and a memory model to reason safely, especially about loads.

---

## 3. Pre‑processing: merging trivial blocks

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2824-L2837
  bool Changed = false;
  bool ShouldContinue = true;

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  // Merge unconditional branches, allowing PRE to catch more
  // optimization opportunities.
  for (BasicBlock &BB : make_early_inc_range(F)) {
    bool RemovedBlock = MergeBlockIntoPredecessor(&BB, &DTU, &LI, MSSAU, MD);
    if (RemovedBlock)
      ++NumGVNBlocks;

    Changed |= RemovedBlock;
  }
  DTU.flush();
```

GVN first does a simple CFG cleanup: it merges blocks with a single predecessor and a trivial unconditional branch into that predecessor, using `MergeBlockIntoPredecessor`.

**Why:**

- Simpler CFG = more GVN opportunities:
  - Fewer PHIs,
  - More straightforward dominance,
  - Better PRE patterns (diamonds without extra trivial blocks).
- It updates dom tree, loop info, MemorySSA, memdep as needed.

---

## 4. Main GVN loop: `iterateOnFunction`

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2839-L2846
  unsigned Iteration = 0;
  while (ShouldContinue) {
    LLVM_DEBUG(dbgs() << "GVN iteration: " << Iteration << "\n");
    (void) Iteration;
    ShouldContinue = iterateOnFunction(F);
    Changed |= ShouldContinue;
    ++Iteration;
  }
```

This is the **GVN fixed point loop**:

- `iterateOnFunction(F)` runs one global sweep.
- If that sweep made any changes (`Changed == true`), GVN tries again, because one CSE or PRE may expose more simplifications.

Let’s look at `iterateOnFunction`.

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L3176-L3188
bool GVNPass::iterateOnFunction(Function &F) {
  cleanupGlobalSets();

  // Top-down walk of the dominator tree.
  bool Changed = false;
  ReversePostOrderTraversal<Function *> RPOT(&F);

  for (BasicBlock *BB : RPOT)
    Changed |= processBlock(BB);

  return Changed;
}
```

Each iteration does:

- `cleanupGlobalSets()`:
  - Clears the value table `VN`, leader table, RPO numbers, implicit control-flow tracking. So each iteration recomputes value numbering from scratch on the transformed IR.
- Walks basic blocks in *reverse postorder* (which is a good approximation of dominance order).
- For each block, calls `processBlock(BB)`.

---

## 5. Per‑block processing: `processBlock`

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2875-L2893
bool GVNPass::processBlock(BasicBlock *BB) {
  if (DeadBlocks.count(BB))
    return false;

  bool ChangedFunction = false;

  // Kill duplicate PHIs created earlier.
  SmallPtrSet<PHINode *, 8> PHINodesToRemove;
  ChangedFunction |= EliminateDuplicatePHINodes(BB, PHINodesToRemove);
  for (PHINode *PN : PHINodesToRemove) {
    removeInstruction(PN);
  }

  // Then process each instruction in order.
  for (Instruction &Inst : make_early_inc_range(*BB))
    ChangedFunction |= processInstruction(&Inst);
  return ChangedFunction;
}
```

So for each block:

- If it’s declared dead (by earlier constant‐prop on branches), skip.
- First, remove trivially duplicate PHIs.
- Then run `processInstruction` for each instruction, in program order.

`processInstruction` is where almost *all* the interesting behavior lives.

---

## 6. Per‑instruction processing: `processInstruction`

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2660-L2801
bool GVNPass::processInstruction(Instruction *I) {
  // 1) Try local simplification
  const DataLayout &DL = I->getDataLayout();
  if (Value *V = simplifyInstruction(I, {DL, TLI, DT, AC})) {
    ...
  }

  // 2) Handle assumes specially
  if (auto *Assume = dyn_cast<AssumeInst>(I))
    return processAssumeIntrinsic(Assume);

  // 3) Handle normal loads (and then masked loads) specially
  if (LoadInst *Load = dyn_cast<LoadInst>(I)) {
    if (processLoad(Load))
      return true;

    unsigned Num = VN.lookupOrAdd(Load);
    LeaderTable.insert(Num, Load, Load->getParent());
    return false;
  }

  if (match(I, m_Intrinsic<Intrinsic::masked_load>()) &&
      processMaskedLoad(cast<IntrinsicInst>(I)))
    return true;

  // 4) Branches: propagate condition values into successors
  if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
    ...
  }

  // 5) Switches: propagate case values
  if (SwitchInst *SI = dyn_cast<SwitchInst>(I)) {
    ...
  }

  // 6) If void-typed, nothing to do for GVN
  if (I->getType()->isVoidTy())
    return false;

  // 7) General value numbering + CSE for non-memory, non-void instructions
  uint32_t NextNum = VN.getNextUnusedValueNumber();
  unsigned Num = VN.lookupOrAdd(I);

  if (isa<AllocaInst>(I) || I->isTerminator() || isa<PHINode>(I)) {
    LeaderTable.insert(Num, I, I->getParent());
    return false;
  }

  if (Num >= NextNum) {
    // New VN: no prior equivalent seen
    LeaderTable.insert(Num, I, I->getParent());
    return false;
  }

  // Try to find dominating leader with same value number
  Value *Repl = findLeader(I->getParent(), Num);
  if (!Repl) {
    LeaderTable.insert(Num, I, I->getParent());
    return false;
  }

  if (Repl == I)
    return false;

  // Found equivalent dominating expression: CSE it away
  patchAndReplaceAllUsesWith(I, Repl);
  if (MD && Repl->getType()->isPtrOrPtrVectorTy())
    MD->invalidateCachedPointerInfo(Repl);
  salvageAndRemoveInstruction(I);
  return true;
}
```

### 6.1. Step 1: local simplification

Before any value numbering, it asks `simplifyInstruction`:

- This uses algebraic simplifications, constant folding, dominator info, etc.
- If it can simplify `I` to some `V`:
  - Replace all uses of `I` with `V`.
  - If `I` becomes dead, remove it.
- This often turns `x & x` into `x`, `x + 0` into `x`, propagates constants, etc.
- It’s cheaper and more powerful locally than value-number based CSE.

**Why:** simpler IR reduces work for the main GVN logic and exposes more CSE opportunities.

### 6.2. Step 2: assume intrinsics → equality propagation

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2093-L2152
bool GVNPass::processAssumeIntrinsic(AssumeInst *IntrinsicI) { ... }
```

For `llvm.assume(cond)`:

- If `cond` is the constant `0`:
  - It inserts a store to poison at that point to mark the code as unreachable (for optimizations that can see it).
  - Possibly removes the assume if it has no bundles.
- If `cond` is some value `V`:
  - It knows `V` is `true` on all paths after the assume, and calls `propagateEquality(V, true, IntrinsicI)`:
    - This is a fairly sophisticated equality propagation that:
      - Replaces dominated uses of `V` with `true`,
      - And derives additional implications (e.g., if `(A == B) == true` then `A` and `B` are equal in that region, etc.).

**Why:** GVN isn’t “just” number-based CSE; it’s also doing logical equality propagation, which improves simplification and further CSE possibilities.

### 6.3. Step 3: loads (local + non‑local) and masked loads

`processLoad` is the heart of the load‑elimination logic:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2159-L2213
bool GVNPass::processLoad(LoadInst *L)
```

- It refuses to touch ordered / volatile / tokenlike loads.
- If the load has no uses → delete it (“dead load elimination”).
- Ask MemoryDependence (`MD->getDependency(L)`) who last affected the memory at that address:

  - If **non‑local**: call `processNonLocalLoad(L)`:
    - Find all non-local dependencies (i.e., blocks where a definition/clobber lives).
    - For each predecessor block, analyze if the load’s value is known there (`AnalyzeLoadAvailability`).
    - If **all** predecessor blocks provide a value: the load is *fully redundant*.
      - Use `ConstructSSAForLoadSet` (SSAUpdater) to synthesize a unified SSA value (maybe with PHI nodes).
      - Replace uses of the load by that value; delete the load.
    - If *some* predecessors provide a value and others don’t:
      - Optionally perform *load PRE*: insert loads in missing predecessors (subject to safety, cost, loops, etc.) and again synthesize SSA to eliminate the original load.
- If **local** dependence:
  - Uses `AnalyzeLoadAvailability` directly on the single dependent instruction:
    - If dependency is a store, load, memset, memcpy, select, or alloca with init, it may forward the stored value, or extract bits from a larger store, or from a memintrinsic, or from a with-overflow, etc.
    - Returns an `AvailableValue` object that knows how to materialize the required load value.
  - If it gets an `AvailableValue`, it materializes the replacement, replaces uses, and deletes the load.

`processMaskedLoad` is a special‑case for `llvm.masked.load` from `llvm.masked.store` with same mask: it replaces the masked load by a `select` of `store_val` vs passthrough.

**Why:** loads are the main expensive memory operations; eliminating them (even when they aren’t syntactically identical) is a big win. GVN’s integration with MemoryDependence/MemorySSA and coercion helpers lets it exploit a lot of aliasing structure.

### 6.4. Step 4–5: branches and switches → equality propagation

For a conditional branch:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2707-L2733
if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
  if (!BI->isConditional())
    return false;

  if (isa<Constant>(BI->getCondition()))
    return processFoldableCondBr(BI);

  Value *BranchCond = BI->getCondition();
  BasicBlock *TrueSucc = BI->getSuccessor(0);
  BasicBlock *FalseSucc = BI->getSuccessor(1);
  ...
  Value *TrueVal = ConstantInt::getTrue(TrueSucc->getContext());
  BasicBlockEdge TrueE(Parent, TrueSucc);
  Changed |= propagateEquality(BranchCond, TrueVal, TrueE);

  Value *FalseVal = ConstantInt::getFalse(FalseSucc->getContext());
  BasicBlockEdge FalseE(Parent, FalseSucc);
  Changed |= propagateEquality(BranchCond, FalseVal, FalseE);
  return Changed;
}
```

- If the condition is a `ConstantInt`, it calls `processFoldableCondBr` (see below) to mark dead regions.
- Otherwise, it knows:
  - On edge Parent → TrueSucc, `cond == true`.
  - On edge Parent → FalseSucc, `cond == false`.
- It calls `propagateEquality` for each edge, which:
  - Replaces dominated uses of `cond` with `true`/`false` along that edge’s region.
  - Deduces further equalities (`(A != B) == false` implies `A == B`, etc.).
  - Adds leaders so future value-numbering sees these equivalences.

For switches, it does the analogous thing per case successor where there is exactly one incoming edge.

**Why:** This is basic conditional constant propagation and predicate reasoning, but integrated with value numbering so later instructions can become constant/folded or CSE’d.

`processFoldableCondBr` also drives control-flow level dead-code marking:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L3312-L3334
bool GVNPass::processFoldableCondBr(BranchInst *BI) {
  ...
  ConstantInt *Cond = dyn_cast<ConstantInt>(BI->getCondition());
  ...
  BasicBlock *DeadRoot =
      Cond->getZExtValue() ? BI->getSuccessor(1) : BI->getSuccessor(0);
  ...
  if (!DeadRoot->getSinglePredecessor())
    DeadRoot = splitCriticalEdges(BI->getParent(), DeadRoot);

  addDeadBlock(DeadRoot);
  return true;
}
```

- Finds the “dead” successor target, splits critical edge if needed, then calls `addDeadBlock` to mark that region dead and poison PHIs from those blocks with poison/undef.

---

### 6.5. Step 7: core value numbering + CSE

If instruction is not covered by special cases, we assign a value number using `VN.lookupOrAdd(I)`. That:

- Builds an `Expression` describing the instruction: opcode, type, and value numbers of operands, plus extra info like comparison predicate, GEP offset decomposition, insert/extract indices, call attributes, and (optionally) memory state when using MemorySSA for loads/stores.
- Hashes that `Expression` to a value number.

Then `processInstruction`:

- Distinguishes newly-seen expressions from already-seen ones using `NextNum`.
- For newly-seen expressions, it just inserts this instruction into `LeaderTable` for that value number.
- If the value number was already in use:
  - It uses `findLeader(BB, Num)` to locate a *dominating* leader with the same value number.
  - If found:
    - It replaces `I` by the dominating value `Repl` and removes `I` (CSE).
  - If not, it just registers `I` as a new leader instance.

**Why:** this is the classic GVN: instructions that compute the same pure expression with equal operands will be collapsed to one representative, as long as dominance holds.

---

## 7. The value numbering data structures

Conceptually:

- `ValueTable VN`:
  - Map `Value* -> uint32_t` (value number).
  - Map `Expression -> value number` (so structurally equal expressions reuse the same number).
  - Additional maps for PHIs and BasicBlocks (for PHI translation and MemorySSA states).
- `LeaderMap LeaderTable`:
  - For each value number, a linked list of `(Value*, BasicBlock*)` entries: the “leaders”.

`findLeader(BB, Num)` picks a leader for that value number that dominates `BB`; constants are preferred (so they are propagated more aggressively).

There is also:

- `phiTranslate` / `phiTranslateImpl`:
  - Used when doing PRE or certain phi-translation cases: given a value number at some `PhiBlock`, and a specific predecessor, try to translate that value number to the value that is live on that predecessor (via PHIs or memory PHIs).
- `areCallValsEqual`:
  - Used to safely treat read-only calls as equal across PHI translation when they provably have no function-local clobbers.

**Why:** Value numbers abstract away *where* a computation occurred. `LeaderTable` ties them back to particular dominating instances, which are the candidates for replacing later uses.

---

## 8. Load PRE and Loop‑Load PRE

Beyond pure GVN, the pass implements:

- **Non-local load elimination** (already covered).
- **Load PRE**: moving a load up into predecessors if its value is partially redundant.
- **Loop load PRE**: a special case where the load is in a loop header and can be hoisted into the preheader while also partially dealing with a single clobbering loop block.

The driver is `processNonLocalLoad` (step 4 in that function):

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2080-L2090
  // Step 4: Eliminate partial redundancy.
  if (!isPREEnabled() || !isLoadPREEnabled())
    return Changed;
  if (!isLoadInLoopPREEnabled() && LI->getLoopFor(Load->getParent()))
    return Changed;

  if (performLoopLoadPRE(Load, ValuesPerBlock, UnavailableBlocks) ||
      PerformLoadPRE(Load, ValuesPerBlock, UnavailableBlocks))
    return true;
```

- `ValuesPerBlock`: for each block, an `AvailableValueInBlock` describing what value we can use for the load there.
- `UnavailableBlocks`: blocks where we cannot get a value.

`PerformLoadPRE` does the general‑case non-loop PRE, with careful guards against:

- Implicit control flow (must not move loads above guards that would deopt instead of fault);
- EH pads;
- Indirect branches;
- Backedge critical edges (unless explicitly allowed);
- Too many predecessors (won’t insert more than one new load by design).

If after everything there is exactly one predecessor where a new load is needed and it’s safe to speculate, it:

- PHI-translates the address into each missing predecessor,
- Creates new loads in those predecessors (or split critical edges),
- Calls `eliminatePartiallyRedundantLoad` which:
  - Optionally also moves identical loads from siblings,
  - Builds SSA with `ConstructSSAForLoadSet`,
  - Replaces the original load’s uses,
  - Deletes the original load.

`performLoopLoadPRE` is like a specialized pattern for loop headers: it wants to hoist the header load to the preheader, while also sinking an additional reload into one specific clobbering loop block when profitable and safe.

**Why:** This is classic PRE for loads: not just eliminating copies that already exist, but moving loads to a better place so the subsequent copy becomes redundant.

---

## 9. Scalar PRE (non‑load expressions)

After the main GVN iterations, if PRE is enabled:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2848-L2857
  if (isPREEnabled()) {
    // Fabricate val-num for dead-code in order to suppress assertion in
    // performPRE().
    assignValNumForDeadCode();
    bool PREChanged = true;
    while (PREChanged) {
      PREChanged = performPRE(F);
      Changed |= PREChanged;
    }
  }
```

`performPRE(F)` walks the function in depth-first order and calls `performScalarPRE` on each instruction.

`performScalarPRE` looks for **diamond patterns**:

- Same computation (same value number) available in some, but not all, predecessors of a block.
- It only does PRE if at most one predecessor is missing the expression.
- It’s careful about:

  - Unsafe speculation (implicit control flow),
  - Crit edges (schedules them for splitting first),
  - Loops backedges,
  - Comparisons and GEPs (excluded for codegen reasons).

If it decides to do PRE:

- If some predecessor `P` lacks the expression:
  - Clone the instruction into `P`, but *rewrite its operands via phiTranslate + findLeader* so it uses values available in `P`.
  - Insert that cloned instruction before the terminator.
  - Value number it and register as a leader.
- Insert a PHI node in the current block that merges either existing or newly inserted computations.
- Replace uses of the original instruction by the PHI and delete the original.

This is PRE for general side‑effect‑free instructions (add, mul, etc.), not just for loads.

**Why:** Normal GVN can only eliminate expressions that are *fully redundant* along the current path; PRE expands the scope to *partially redundant* expressions by sinking/hoisting them to a single join point.

---

## 10. Cleanup, dead blocks, and bookkeeping

At the end of `runImpl`:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/b99970d929c0da9cdaa7a82f16197b07c00e919a/llvm/lib/Transforms/Scalar/GVN.cpp#L2864-L2873
  cleanupGlobalSets();
  DeadBlocks.clear();

  if (MSSA && VerifyMemorySSA)
    MSSA->verifyMemorySSA();

  return Changed;
}
```

- `cleanupGlobalSets()` clears VN/LeaderTable/ICF etc.
- `DeadBlocks` is where `processFoldableCondBr/addDeadBlock` marked blocks as dead earlier; it’s now reset.
- Optionally verifies MemorySSA consistency.

`removeInstruction` ensures instructions are dropped from all internal structures (VN, LeaderTable, MemorySSA, MemDep) before erasing them from IR.

`addDeadBlock` and `processFoldableCondBr` maintain a set of blocks known to be dead due to constant branches and:

- Propagate deadness through the dominator tree,
- Update PHIs at the dominance frontier to use poison for dead predecessors,
- Split critical edges when needed.

**Why:** Once you know a region is unreachable, many values there are “don’t care”. Poisoning them in PHIs can unlock more simplifications downstream (sinks of those PHIs may get removed, etc.), and skipping those blocks in GVN avoids useless work.

---

## 11. Summary: conceptual flow from `runImpl`

Putting it all together “execution-wise”:

1. `runImpl` initializes analysis handles and MemorySSA updater.
2. It merges trivial blocks to simplify the CFG.
3. Then, repeatedly:
   - Clears all GVN internal state.
   - Walks the function in RPO:
     - For each block not in `DeadBlocks`:
       - Removes duplicate PHIs.
       - For each instruction:
         - Try local simplification first.
         - Special-case: handles `assume`, loads/masked loads (with memdep/MSSA), branches, switches.
         - Otherwise, value-numbers it and either:
           - Registers it as a new leader, or
           - Finds a dominating leader with the same value number and CSEs it away.
         - Branches and switches also drive equality propagation and dead-block discovery.
   - If this iteration changed the IR, repeat.
4. If PRE is enabled:
   - Fabricate value numbers for dead code so PRE’s value-number queries won’t assert.
   - Repeatedly perform **scalar PRE** over the function until it stops changing:
     - For each instruction, detect partial redundancy across predecessors.
     - Where safe and cost-effective, insert clones in missing predecessors and PHIs at joins, then remove the redundant original.
   - Also use load PRE and loop load PRE (triggered from `processNonLocalLoad`).

5. Clear internal tables and return whether the function changed.

## how `ValueTable::createGEPExpr` makes GEPs with different type encodings hash to the same expression
`ValueTable::createGEPExpr` throws away the *typed* indexing (struct indices vs array indices, element types etc.) and instead computes a *pure byte offset* from the base pointer. Two GEPs that compute the same base + byte offset will then produce the same `Expression`, even if they use different type encodings.
### 1. The `Expression` shape (what GEPs are compared on)
First, recall what an `Expression` looks like:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/1383dd8dbe7a667cbac98554e80b261cb86c86ee/llvm/lib/Transforms/Scalar/GVN.cpp#L117-L143
struct llvm::GVNPass::Expression {
  uint32_t Opcode;
  bool Commutative = false;
  // The type is not necessarily the result type of the expression, it may be
  // any additional type needed to disambiguate the expression.
  Type *Ty = nullptr;
  SmallVector<uint32_t, 4> VarArgs;

  AttributeList Attrs;

  Expression(uint32_t Op = ~2U) : Opcode(Op) {}

  bool operator==(const Expression &Other) const {
    if (Opcode != Other.Opcode)
      return false;
    if (Opcode == ~0U || Opcode == ~1U)
      return true;
    if (Ty != Other.Ty)
      return false;
    if (VarArgs != Other.VarArgs)
      return false;
    if ((!Attrs.isEmpty() || !Other.Attrs.isEmpty()) &&
        !Attrs.intersectWith(Ty->getContext(), Other.Attrs).has_value())
      return false;
    return true;
  }

  friend hash_code hash_value(const Expression &Value) {
    return hash_combine(Value.Opcode, Value.Ty,
                        hash_combine_range(Value.VarArgs));
  }
};
```

For a GEP-expression, `Opcode` will be `Instruction::GetElementPtr` and `VarArgs` will encode:

- The value number of the base pointer, and
- A canonicalized “offset representation” derived from indices and pointee layout.

So the magic is in how `createGEPExpr` *fills* `VarArgs`.

### 2. `ValueTable::createGEPExpr`

Here is the implementation (trimmed to the essential bits):

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/1383dd8dbe7a667cbac98554e80b261cb86c86ee/llvm/lib/Transforms/Scalar/GVN.cpp#L556-L640
GVNPass::Expression ValueTable::createGEPExpr(GetElementPtrInst *GEP) {
  SmallVector<Value *, 4> Ops(GEP->getNumOperands());
  std::copy(GEP->op_begin(), GEP->op_end(), Ops.begin());

  // Get the base pointer; we may strip bitcasts etc.
  Value *Base = Ops[0];

  // Try to turn the GEP into base + byte-offset, ignoring types.
  APInt Offset(DL.getPointerTypeSizeInBits(Base->getType()), 0);
  bool Success = GEP->accumulateConstantOffset(DL, Offset);

  Expression E(Instruction::GetElementPtr);
  E.Ty = GEP->getType()->getScalarType();

  // First var-arg is the value number of the base pointer.
  E.VarArgs.push_back(lookupOrAdd(Base));

  if (!Success) {
    // If we can't compute a constant offset, fall back to per-index VNs.
    for (unsigned i = 1, e = Ops.size(); i != e; ++i)
      E.VarArgs.push_back(lookupOrAdd(Ops[i]));
    return E;
  }

  // If we have a constant offset, encode it directly as an integer in bytes.
  // Normalize to byte-offset so different type encodings that give the same
  // address compute the same expression.
  uint64_t ByteOffset = (Offset.sextOrTrunc(64)).getZExtValue();
  E.VarArgs.push_back(ByteOffset & 0xFFFFFFFFu);
  E.VarArgs.push_back((ByteOffset >> 32) & 0xFFFFFFFFu);

  return E;
}
```

High-level behavior:

1. Collect all operands of the GEP.
2. Take operand 0 as `Base`.
3. Ask `GEP->accumulateConstantOffset(DL, Offset)` to compute a *constant* byte offset from `Base` using the target `DataLayout`.
4. Always include the value number of `Base` as the first entry in `VarArgs`.
5. If constant-offset computation failed:
   - Fall back to recording VN of each index operand individually (type-sensitive form).
6. If it succeeded:
   - Convert the final offset to a 64-bit integer and pack it into two 32-bit chunks in `VarArgs`.

So two GEPs will compare equal if:

- They have the same base pointer value number, and
- They have the same computed constant byte offset, regardless of how the indices and pointee types were written.

### 3. Where the type-encoding is removed: `accumulateConstantOffset`

GEP’s method:

```c++ name=llvm/include/llvm/IR/GetElementPtrTypeIterator.h url=https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Operator.h#L599-L638
bool GEPOperator::accumulateConstantOffset(const DataLayout &DL,
                                           APInt &Offset) const;
```

Conceptually, `accumulateConstantOffset` does:

- Walk the GEP indices and the pointee type:
  - For a struct index `i`, add the struct-field offset from `DL.getStructLayout`.
  - For an array/pointer index `i`, add `i * elementSize` (element size from `DL.getTypeAllocSize`).
- If it encounters any non-constant index or something it can’t reduce, it returns `false`.
- If successful, `Offset` is the **total byte offset** from the base.

The important point: the same address computation written via different types, like:

```llvm
; GEP #1: via [10 x i32]*
%p1 = getelementptr [10 x i32], [10 x i32]* %base, i64 0, i64 3

; GEP #2: via i8*
%p2 = getelementptr i8, i8* %base_i8, i64 12
```

If `%base` and `%base_i8` are the same underlying pointer (ignoring bitcasts), both GEPs produce offset 12 bytes from base. `accumulateConstantOffset` will compute:

- For GEP #1: `3 * 4 = 12`
- For GEP #2: `12 * 1 = 12`

Thus both end up with the same `ByteOffset` and the same base VN, so `createGEPExpr` returns identical `Expression`s.

### 4. Why this normalisation matters for GVN

Without this trick, `GVN` would see GEPs as structurally different because:

- One has two indices `[0, 3]` on `[10 x i32]*`.
- Another has one index `[12]` on `i8*`.

Typed GEP encodings would not hash equal structurally, so the expressions would end up with different value numbers, and GVN wouldn’t CSE them.

By canonicalizing to:

- `VN(base)`, and
- `byte_offset`,

GVN can:

- Recognize equivalent address computations written through different type paths.
- CSE loads/stores and other memory ops that use those pointers.
- Combine this with MemorySSA/memdep for stronger load/store optimization.

If `accumulateConstantOffset` fails (non-constant indices or too complex typing), it *falls back* to the conservative representation that includes value numbers of each index operand, so no unsafe merging happens.

## how load availability + `AvailableValue`/`AvailableValueInBlock` are used with `SSAUpdater` 

At a high level:

- `AnalyzeLoadAvailability` tells GVN, *for each predecessor block*, whether the value of a given load is already known there, and if so, how to materialize it (as an `AvailableValue` / `AvailableValueInBlock`).
- `PerformLoadPRE` / `performLoopLoadPRE` decide whether to do PRE based on that availability info.
- If they decide “yes”, they:
  - Build a per-block map `AvailableLoads: BasicBlock* -> Value*` describing the value of the load in every relevant predecessor.
  - Call `eliminatePartiallyRedundantLoad`, which:
    - Uses `SSAUpdater` to form SSA over those `AvailableLoads`.
    - Gets a single SSA value `Vphi` at the original load’s block.
    - Replaces uses of the original load with `Vphi` and erases the load.

So `SSAUpdater` is the thing that turns the *scattered*, per-block `AvailableValue` instances into a *single* SSA value with PHIs as needed.

## 1. From memdep to `AvailableValueInBlock`

First, for a non-local load, GVN gathers memory-dependence info:

- It asks MemoryDependence / MemorySSA for non-local dependencies of load `L`.
- It calls:

```c++ name=llvm/include/llvm/Transforms/Scalar/GVN.h url=https://github.com/llvm/llvm-project/blob/379d95c210aa6ebca885f7352bb4030063b7c3d6/llvm/include/llvm/Transforms/Scalar/GVN.h#L351-L360
/// Given a list of non-local dependencies, determine if a value is
/// available for the load in each specified block.  If it is, add it to
/// ValuesPerBlock.  If not, add it to UnavailableBlocks.
void AnalyzeLoadAvailability(LoadInst *Load, LoadDepVect &Deps,
                             AvailValInBlkVect &ValuesPerBlock,
                             UnavailBlkVect &UnavailableBlocks);
```

This fills:

- `ValuesPerBlock`: a vector of `AvailableValueInBlock` records.
- `UnavailableBlocks`: blocks where we *don’t* know the load’s value.

`AvailableValueInBlock` (defined earlier in `GVN.cpp`) has essentially:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/379d95c210aa6ebca885f7352bb4030063b7c3d6/llvm/lib/Transforms/Scalar/GVN.cpp#L197-L234
namespace llvm::gvn {

struct AvailableValueInBlock {
  BasicBlock *BB;
  AvailableValue AV;   // describes how to materialize the value in this BB
};
```

and `AvailableValue` itself is an enum + payload describing kinds of materializable values (simple direct value, value loaded from memory, value from memcpy/memset, etc.):

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/379d95c210aa6ebca885f7352bb4030063b7c3d6/llvm/lib/Transforms/Scalar/GVN.cpp#L196-L235
struct llvm::gvn::AvailableValue {
  enum class ValType {
    SimpleVal, // A simple offsetted value that is accessed.
    LoadVal,   // A value produced by a load.
    MemIntrin, // A value produced by a memset/memcpy/memmove.
    // ...
  } ValType;
  Value *V;    // or more complex payload...

  Value *materialize(IRBuilderBase &Builder) const;
};
```

So for each block `BB` where the load’s value is known, GVN stores:

- `BB` – the block.
- `AV` – enough info to “replay” the load’s value in that block (often just a `Value*`).

---

## 2. Deciding to do PRE: `PerformLoadPRE`

When `processNonLocalLoad` considers PRE, it eventually calls:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/379d95c210aa6ebca885f7352bb4030063b7c3d6/llvm/lib/Transforms/Scalar/GVN.cpp#L1657-L1710
bool GVNPass::PerformLoadPRE(LoadInst *Load, AvailValInBlkVect &ValuesPerBlock,
                             UnavailBlkVect &UnavailableBlocks) {
  // ... safety checks, walking up single-predecessor chain, etc. ...

  // Check to see how many predecessors have the loaded value fully
  // available.
  MapVector<BasicBlock *, Value *> PredLoads;
  DenseMap<BasicBlock *, AvailabilityState> FullyAvailableBlocks;
  for (const AvailableValueInBlock &AV : ValuesPerBlock)
    FullyAvailableBlocks[AV.BB] = AvailabilityState::Available;
  for (BasicBlock *UnavailableBB : UnavailableBlocks)
    FullyAvailableBlocks[UnavailableBB] = AvailabilityState::Unavailable;

  // ... now inspect predecessors of LoadBB, see how many need insertion, etc. ...
}
```

Here:

- `ValuesPerBlock` is the output from `AnalyzeLoadAvailability`.
- `FullyAvailableBlocks` is a map `BB -> {Available, Unavailable}`.
- As it walks predecessors of `LoadBB`, it classifies them into:
  - Those where we already have an available value (from `ValuesPerBlock`).
  - Those where we’d need to insert a new load (from `UnavailableBlocks`).
- It only proceeds if *at most one* predecessor needs a new load (to avoid code-size growth).

It then builds:

- `PredLoads: MapVector<BasicBlock*, Value*>`:
  - For each predecessor `Pred`:
    - If the load’s value is already available there, it materializes it using the `AvailableValue`.
    - Otherwise (for the one missing pred), it will schedule a new load to be created there (possibly splitting a critical edge), and adds that new load to `PredLoads`.

So at the end of `PerformLoadPRE`’s classification, you conceptually have:

- A set of predecessors `Pred_i` of `LoadBB`.
- For each, a `Value*` `PredLoads[Pred_i]` describing the load’s value along that edge.

`PerformLoadPRE` then calls:

```c++
eliminatePartiallyRedundantLoad(Load, ValuesPerBlock, PredLoads,
                                &CriticalEdgePredAndLoad);
```

This is where `SSAUpdater` comes in.


## 3. Stitching it into SSA: `eliminatePartiallyRedundantLoad`

`eliminatePartiallyRedundantLoad` is responsible for:

- Using an SSA construction to connect all `PredLoads` into a single SSA value in `LoadBB`.
- Replacing the original load’s uses by that SSA value.
- Deleting the original load.

The structure (simplified) looks like:

```c++ name=llvm/lib/Transforms/Scalar/GVN.cpp url=https://github.com/llvm/llvm-project/blob/379d95c210aa6ebca885f7352bb4030063b7c3d6/llvm/lib/Transforms/Scalar/GVN.cpp#L1728-L1795
void GVNPass::eliminatePartiallyRedundantLoad(
    LoadInst *Load, AvailValInBlkVect &ValuesPerBlock,
    MapVector<BasicBlock *, Value *> &AvailableLoads,
    MapVector<BasicBlock *, LoadInst *> *CriticalEdgePredAndLoad) {

  // 1. For blocks where we know the value is available but don't yet have a
  //    concrete Value*, materialize them.
  IRBuilder<> Builder(Load->getContext());
  for (AvailableValueInBlock &AV : ValuesPerBlock) {
    if (AvailableLoads.count(AV.BB))
      continue;
    Builder.SetInsertPoint(AV.BB->getTerminator());
    Value *V = AV.AV.materialize(Builder);  // uses AvailableValue::materialize
    AvailableLoads[AV.BB] = V;
  }

  // 2. Build SSA form for the load value reaching LoadBB.
  SSAUpdater SSA(&InsertedPHIs);
  SSA.Initialize(Load->getType(), Load->getName());

  for (auto &Entry : AvailableLoads)
    SSA.AddAvailableValue(Entry.first, Entry.second);

  Value *NewVal = SSA.GetValueInMiddleOfBlock(Load->getParent());

  // 3. Replace and erase the original load.
  Load->replaceAllUsesWith(NewVal);
  salvageAndRemoveInstruction(Load);
}
```

Key points:

1. **Materializing `AvailableValue`s**  
   For each `AvailableValueInBlock AV`:

   - If `AvailableLoads[AV.BB]` is already set (maybe from a real existing load or a hoisted load), we skip.
   - Otherwise, we *materialize* the value at the end of that block using `AvailableValue::materialize`:

     - For a `SimpleVal` this might just be a `bitcast` or GEP.
     - For `LoadVal` it might be a cloned load with the right address.
     - For `MemIntrin`, it may insert `load`s from a `memcpy` destination, etc.

   - We then record that concrete `Value*` in `AvailableLoads[AV.BB]`.

   After this loop, for every block in `ValuesPerBlock`, `AvailableLoads` has a real `Value*` that is live at the end of that block and equal to the logical load.

2. **Setting up the SSAUpdater**  

   ```c++
   SSAUpdater SSA(&InsertedPHIs);
   SSA.Initialize(Load->getType(), Load->getName());
   for (auto &Entry : AvailableLoads)
     SSA.AddAvailableValue(Entry.first, Entry.second);
   ```

   - `SSA.Initialize` records the type and base name.
   - `AddAvailableValue(BB, V)` says: “in block `BB`, at the end, the value of our symbolic variable is `V`”.

   At this point, `SSAUpdater` knows all the defining blocks and their values.

3. **Computing the value in the original block**  

   ```c++
   Value *NewVal = SSA.GetValueInMiddleOfBlock(Load->getParent());
   ```

   - This asks: “What is the correct SSA value representing this variable at the location of the `Load` in its block?”
   - `SSAUpdater` will:
     - If there’s a unique dominating definition that reaches here, just return that `Value*`.
     - Otherwise, it inserts one or more PHI nodes in `LoadBB` (and potentially in other join blocks up the dominator tree) to merge the mismatched incoming values.
     - It records those PHIs both internally and in `InsertedPHIs` vector.

   So effectively, `SSAUpdater` reconstructs the SSA form “as if” the load had been computed in each predecessor and then merged via PHIs.

4. **Replacing the original load**  

   ```c++
   Load->replaceAllUsesWith(NewVal);
   salvageAndRemoveInstruction(Load);
   ```

   - Users of the original load now see the new SSA value, which is either:
     - A single dominating load (if all preds had the same value), or
     - A PHI node combining multiple loads/values.

## 4. How `AvailableValue` interacts with SSAUpdater

Conceptually:

- `AvailableValue` is a *recipe* for “what is the load’s value in block BB?”
- Before `SSAUpdater` is used, `eliminatePartiallyRedundantLoad` turns all those recipes into actual IR instructions (`Value*`) using `materialize`.
- `SSAUpdater` doesn’t know anything about loads or memory; it just knows:

  - There is a variable `X`,
  - It has definitions in blocks `{B1, B2, ..., Bk}`,
  - With values `{V1, V2, ..., Vk}`,
  - Please give me `X` at some use location.

So the pipeline is:

1. **Dep analysis → `AvailableValue` / `AvailableValueInBlock`**  
   - Use memdep/MSSA to figure out what defines the memory contents.
   - Use `AnalyzeLoadAvailability` to abstract these definitions as `AvailableValue`s.

2. **PRE decision (`PerformLoadPRE` / `performLoopLoadPRE`)**  
   - Decide whether to move/insert loads into missing preds.
   - Build `AvailableLoads: BB -> Value*` for all preds.

3. **SSA construction (`eliminatePartiallyRedundantLoad` + `SSAUpdater`)**  
   - Materialize `AvailableValue`s where needed.
   - Feed all `(BB, Value*)` into `SSAUpdater`.
   - Ask `SSAUpdater` to produce `NewVal` in the load’s block (with PHIs if needed).
   - Replace original load with `NewVal`.


## 5. Intuitive example

Imagine:

```llvm
; Pred1:
  %x1 = load i32, i32* %p
  br label %join

; Pred2:
  ; no load yet
  br label %join

join:
  %x = load i32, i32* %p
  use(%x)
```

Analysis says:

- In `Pred1`, load value is already available via `%x1` (`AvailableValueInBlock{BB=Pred1, AV=SimpleVal(%x1)}`).
- In `Pred2`, it’s not available; `PerformLoadPRE` decides it’s OK to insert a new load:

```llvm
Pred2:
  %x2 = load i32, i32* %p     ; new
  br label %join
```

Now:

- `AvailableLoads[Pred1] = %x1`
- `AvailableLoads[Pred2] = %x2`

`eliminatePartiallyRedundantLoad`:

- Calls `SSA.Initialize(i32, "%x")`.
- Adds the two available values.
- `SSA.GetValueInMiddleOfBlock(join)` inserts:

```llvm
join:
  %x.phi = phi i32 [ %x1, %Pred1 ], [ %x2, %Pred2 ]
  use(%x.phi)
```

Then it removes the original `%x = load...`.

Result: the original join-load is eliminated; its uses see a PHI merging `%x1` and `%x2`.

All of this is generic SSA construction handled by `SSAUpdater`. GVN’s domain-specific job is to:

- Determine where values are available (`AvailableValueInBlock`), and
- Where new computations (loads) should be inserted to make the expression fully/partially redundant.

