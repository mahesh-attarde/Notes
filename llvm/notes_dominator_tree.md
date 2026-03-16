# LLVM Dominator Trees (IR & MIR)
## 1. Concept: 
In a control-flow graph (CFG):

- A basic block `D` **dominates** a basic block `B` if every path from the entry block to `B` goes through `D`.
- The entry block of a function dominates all blocks.
- A **strict dominator** of `B` is a dominator `D` where `D != B`.
- The **immediate dominator** (`IDom`) of `B` is the unique strict dominator of `B` that does not strictly dominate any other strict dominator of `B`.
- The **dominator tree** is a tree whose:
  - root is the entry block, and
  - parent of each block is its immediate dominator.

Dominator trees are used for:
- SSA construction (placing φ-nodes),
- various optimizations (LICM, GVN, etc.),
- reasoning about control-flow and safety of transformations.

LLVM builds and maintains these structures both for:
- **IR** (`llvm::DominatorTree`, `llvm::PostDominatorTree`) and
- **MIR** (`llvm::MachineDominatorTree`, `llvm::MachinePostDominatorTree`).

## 2. IR-Level Dominator Tree

### 2.1 Main Classes

- `llvm::DominatorTree`  Represents forward dominators in an IR function.
- `llvm::DominatorTreeWrapperPass` (legacy pass manager)
- `llvm::DominatorTreeAnalysis` / `llvm::DominatorTreePrinterPass` (new PM)
Related ADT
- `llvm::DominatorTreeBase<BasicBlock>`
- `llvm::DomTreeNode` – nodes in the dom tree.
- `llvm::PostDominatorTree` – for post-dominators.

### 2.2 Core API (IR)
Commonly used methods on `llvm::DominatorTree`:
- Querying dominance:
  - `bool dominates(const BasicBlock *A, const BasicBlock *B) const;`
  - `bool dominates(const Instruction *Def, const Instruction *Use) const;`
  - `bool properlyDominates(const BasicBlock *A, const BasicBlock *B) const;`
- Immediate dominator:
  - `BasicBlock *getRoot() const;`  
  - `BasicBlock *getNode(BasicBlock *BB);` via `DomTreeNode *`
  - `BasicBlock *getIDom(BasicBlock *BB) const;` (or via node: `Node->getIDom()`)
- Tree navigation:
  - `DomTreeNode *getNode(BasicBlock *BB) const;`
  - `DomTreeNode *getRootNode() const;`
  - On `DomTreeNode`:
    - `BasicBlock *getBlock() const;`
    - `DomTreeNode *getIDom() const;`
    - `const std::vector<DomTreeNode *> &getChildren() const;`
- Updates:
  - `void recalculate(Function &F);`
  - Incremental updates: `insertEdge`, `eraseEdge`, `addNewBlock`, `eraseBlock`, etc.

### 2.3 Example: Using DominatorTree in an IR Function Pass (Legacy PM)

```cpp name=IRDomTreeExample.cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct IRDomTreeExample : public FunctionPass {
  static char ID;
  IRDomTreeExample() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

    errs() << "Function: " << F.getName() << "\n";
    for (auto &BB : F) {
      errs() << " BasicBlock: " << BB.getName() << "\n";

      DomTreeNode *Node = DT.getNode(&BB);
      if (!Node) continue;

      if (auto *IDom = Node->getIDom()) {
        errs() << "  IDom: " << IDom->getBlock()->getName() << "\n";
      } else {
        errs() << "  IDom: (none, likely entry)\n";
      }

      errs() << "  Children (blocks immediately dominated by this block):\n";
      for (DomTreeNode *Child : Node->getChildren()) {
        errs() << "    " << Child->getBlock()->getName() << "\n";
      }
    }
    return false; // analysis only
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesAll();
  }
};
} // end anonymous namespace

char IRDomTreeExample::ID = 0;
static RegisterPass<IRDomTreeExample>
    X("ir-dom-tree-example", "IR Dominator Tree Example", false, true);
```

### 2.4 Example: Checking Instruction Dominance

```cpp name=IRInstructionDominance.cpp
bool dominatesAllUses(Value *V, DominatorTree &DT) {
  if (Instruction *DefI = dyn_cast<Instruction>(V)) {
    for (User *U : V->users()) {
      if (Instruction *UseI = dyn_cast<Instruction>(U)) {
        if (!DT.dominates(DefI, UseI))
          return false;
      }
    }
  }
  return true;
}
```

## 3. MIR-Level Dominator Tree

At the Machine IR level, LLVM uses analogous concepts but with `MachineBasicBlock` and `MachineFunction`.

### 3.1 Main Classes

- `llvm::MachineDominatorTree`  
  - Dominator tree over `MachineBasicBlock`s.
- `llvm::MachinePostDominatorTree`
- `llvm::MachineDomTreeNode` (typedef for `DomTreeNodeBase<MachineBasicBlock>`)

Headers:

- `#include "llvm/CodeGen/MachineDominators.h"`

### 3.2 Core API (MIR)

`MachineDominatorTree` mirrors `DominatorTree` with Machine-level types:

- Querying dominance:
  - `bool dominates(const MachineBasicBlock *A, const MachineBasicBlock *B) const;`
  - `bool dominates(const MachineInstr *Def, const MachineInstr *Use) const;`
- Immediate dominator:
  - `MachineBasicBlock *getRoot() const;`
  - `MachineDomTreeNode *getNode(MachineBasicBlock *BB) const;`
  - `MachineBasicBlock *getIDom(MachineBasicBlock *BB) const;`
- Tree navigation:
  - On `MachineDomTreeNode`:
    - `MachineBasicBlock *getBlock() const;`
    - `MachineDomTreeNode *getIDom() const;`
    - `const std::vector<MachineDomTreeNode *> &getChildren() const;`


### 3.3 Example: Using MachineDominatorTree in a MachineFunctionPass

```cpp name=MIRDomTreeExample.cpp
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct MIRDomTreeExample : public MachineFunctionPass {
  static char ID;
  MIRDomTreeExample() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();

    errs() << "MachineFunction: " << MF.getName() << "\n";
    for (MachineBasicBlock &MBB : MF) {
      errs() << " MBB: " << MBB.getNumber() << " ";
      if (MBB.hasName())
        errs() << "(" << MBB.getName() << ")";
      errs() << "\n";

      MachineDomTreeNode *Node = MDT.getNode(&MBB);
      if (!Node) continue;

      if (auto *IDom = Node->getIDom()) {
        errs() << "  IDom: " << IDom->getBlock()->getNumber() << "\n";
      } else {
        errs() << "  IDom: (none, likely entry)\n";
      }

      errs() << "  Children:\n";
      for (MachineDomTreeNode *Child : Node->getChildren()) {
        errs() << "    " << Child->getBlock()->getNumber() << "\n";
      }
    }
    return false; // analysis only
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.setPreservesAll();
  }
};
} // end anonymous namespace

char MIRDomTreeExample::ID = 0;
static RegisterPass<MIRDomTreeExample>
    Z("mir-dom-tree-example", "MIR Dominator Tree Example", false, true);
```
- Use LLVM’s existing passes as examples:
  - IR: `-view-dom` / `-view-cfg-only` and passes like `-mem2reg`, `-licm`.
  - MIR: `-machine-dom-info` related passes.
- Read:
  - `llvm/include/llvm/IR/Dominators.h`
  - `llvm/include/llvm/CodeGen/MachineDominators.h`
  - Implementations in `lib/IR/` and `lib/CodeGen/`.

# How LICM and MachineLICM Use the Dominator Tree

## 1. LICM (IR-Level) and DominatorTree

### 1.1 What LICM Does (IR)

LICM (Loop Invariant Code Motion) moves loop-invariant and safe-to-execute code:

- **Hoisting**: from inside a loop to a **preheader** (before loop body).
- **Sinking**: from loop body to **latch/exit** blocks when legal.

To do that safely, LICM needs to know:

1. If the definition **dominates** all its uses.
2. If the transformed control flow preserves semantics (e.g., no new paths where an instruction is executed).

### 1.2 Key Analyses Used by LICM

IR LICM typically uses:

- `LoopInfo` / `Loop`
- `DominatorTree` (`llvm::DominatorTree`)
- `AliasAnalysis` / `MemorySSA` etc.

The dominator tree is essential in multiple checks.

### 1.3 Typical DominatorTree Uses in LICM

#### A. Checking That a Definition Dominates All Uses

Before hoisting an instruction `I` out of a loop:

- LICM needs to ensure that **after** hoisting, all uses are still dominated by `I`.

Canonical pattern:

```cpp name=IR_LICM_DominatesUses.cpp
bool dominatesAllUses(Instruction *I,
                      DominatorTree &DT,
                      const Loop *L) {
  for (User *U : I->users()) {
    auto *UseI = dyn_cast<Instruction>(U);
    if (!UseI)
      continue;
    // Only care about uses within the loop
    if (!L->contains(UseI->getParent()))
      continue;

    // If I does not dominate the use, LICM can't just hoist as-is
    if (!DT.dominates(I, UseI))
      return false;
  }
  return true;
}
```

LICM uses this kind of reasoning to check:

- For **hoisting**: The (potential) new position will dominate all uses.
- For **sinking**: All remaining uses are in blocks dominated by a candidate sink block.

#### B. Ensuring Safe Hoisting Position (Preheader)

To hoist `I` into the loop **preheader**:

- LICM verifies:
  - The loop has a unique preheader.
  - The preheader **dominates** all loop blocks that contain uses.

Conceptually:

```cpp name=IR_LICM_HoistCheck.cpp
bool canHoistToPreheader(Instruction *I,
                         DominatorTree &DT,
                         Loop *L) {
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader)
    return false;

  // Check: preheader dominates every block where I is used (inside the loop).
  for (User *U : I->users()) {
    auto *UseI = dyn_cast<Instruction>(U);
    if (!UseI || !L->contains(UseI->getParent()))
      continue;
    if (!DT.dominates(Preheader, UseI->getParent()))
      return false;
  }

  return true;
}
```

#### C. Sinking to Latch/Exit Blocks

For **sinking** an instruction:

- LICM may move it to a block that dominates all remaining uses but is as late as possible (e.g., loop latch or some exit block).
- It uses `DominatorTree` to find a block that:
  - Is **dominated by** the original block, and
  - **dominates** all use blocks.

The algorithm effectively intersects dominator relationships of all use blocks.

#### D. Reasoning About Control-Dependence

Dominator information is indirectly used to ensure:

- Hoisting doesn’t move an instruction above a point where it might become **conditionally executed**, or
- Introduce execution on paths where it wasn’t executed before (e.g., speculative execution safety).

For example:

- `I` must be **guaranteed to execute** along all paths that reach the hoist point.
- The block of `I` must be dominated by the **header** and not control-dependent on inner conditions that don’t dominate the preheader.

## 2. MachineLICM (MIR-Level) and MachineDominatorTree

At the Machine IR level, `MachineLICM` performs a similar optimization, but:

- Works on `MachineInstr`, `MachineBasicBlock`, `MachineFunction`.
- Must respect low-level constraints: register allocation, scheduling, hardware hazards, etc.

### 2.1 Analyses Used

MachineLICM often relies on:

- `MachineLoopInfo` / `MachineLoop`
- `MachineDominatorTree` (`llvm::MachineDominatorTree`)
- `MachineBlockFrequencyInfo` (for profitability)
- Target-specific info, alias/memory info, etc.

### 2.2 Typical MachineDominatorTree Uses

The patterns are **analogous** to IR LICM, but with machine types.

#### A. Dominance Checks for Instructions

Machine-level version of “definition dominates all uses”:

```cpp name=MIR_LICM_DominatesUses.cpp
bool dominatesAllMachineUses(MachineInstr *MI,
                             MachineDominatorTree &MDT,
                             MachineLoop *ML) {
  for (auto &MO : MI->defs()) {
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;

    for (auto UI = MRI.use_begin(Reg), UE = MRI.use_end(); UI != UE; ++UI) {
      MachineInstr &UseMI = *UI->getParent();
      if (!ML->contains(UseMI.getParent()))
        continue;

      if (!MDT.dominates(MI, &UseMI))
        return false;
    }
  }
  return true;
}
```
It is Code Pattern.

#### B. Hoisting to Machine Loop Preheader (Prolog Block)

MachineLICM:

- Identifies a **preheader-like** block for a machine loop (could be a dedicated preheader or a block that uniquely flows into the loop header).
- Uses `MachineDominatorTree` to verify that:

  - The chosen hoist block **dominates** the loop header and relevant blocks, and  
  - After hoisting, the instruction will still dominate all its uses.

Conceptually:

```cpp name=MIR_LICM_HoistCheck.cpp
bool canHoistMITo(MachineInstr *MI,
                  MachineBasicBlock *HoistMBB,
                  MachineDominatorTree &MDT,
                  MachineLoop *ML) {
  for (MachineOperand &MO : MI->operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;

    for (MachineInstr &UseMI : MRI.use_instructions(Reg)) {
      MachineBasicBlock *UseMBB = UseMI.getParent();
      if (!ML->contains(UseMBB))
        continue;

      if (!MDT.dominates(HoistMBB, UseMBB))
        return false;
    }
  }

  return true;
}
```

#### C. Sinking to Latch or Exit MachineBasicBlocks

Similar to IR:

- MachineLICM may sink instructions into latches or exit blocks when:
  - Those blocks are dominated by the instruction’s original block.
  - They **dominate** all uses.

`MachineDominatorTree` provides:

- `dominates(MachineBasicBlock *A, MachineBasicBlock *B)`
- `getNode`, `getIDom`, etc.  

MachineLICM combines these to pick a “lowest” safe sink point.

#### D. Loop Structure and Nesting

Machine loops and their headers/latches:

- Are understood with `MachineLoopInfo`.
- But `MachineLoopInfo` itself depends on a consistent CFG and often on dominator tree updates.
- In transformations that change CFG, MachineLICM or helper utilities may update `MachineDominatorTree` incrementally (`MDT.insertEdge`, `MDT.eraseEdge`, etc.) to preserve correctness for subsequent dominance queries.

> Use the dominator tree to prove that moving an instruction (hoist or sink) does not change which paths execute that instruction relative to its uses and side effects.

 - `lib/Transforms/Scalar/LICM.cpp`  
 - `lib/CodeGen/MachineLICM.cpp`  
