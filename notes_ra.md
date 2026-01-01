# LLVM Register Allocation Vocabulary
vocabulary needed for debugging/reading RA, plus important RA-adjacent pieces that explain *where values go* (VRM/MFI/FI), *why copies appear* (two-address, PHIs), and *why GPU RA behaves differently* (occupancy, SGPR/VGPR).
---

## 0) Pipeline Context
**Typical flow (conceptual):**
1. IR → (SelectionDAG or GlobalISel) → **Machine IR (MIR)** with **vregs**
2. Pre-RA cleanup + **phi elimination** / **two-address** handling / copy insertion
3. Liveness computed (**LiveIntervals**, **SlotIndexes**)
4. Register allocation (e.g., **RegAllocGreedy** or **RegAllocFast**)
5. Spilling/reloading inserted by a **Spiller**
6. MIR rewritten to physregs (**VirtRegMap** applied)
7. Post-RA fixups, scheduling, prologue/epilogue insertion, final emit

---

## A) Main Lingo

### 1) **MachineRegisterInfo (MRI)**
**What**: Tracks all vregs, their regclasses/banks, def-use chains, and allocation hints.  
**Why RA cares**: RA queries MRI to know constraints and uses/defs.

### 2) **TargetRegisterInfo (TRI)** + **TargetInstrInfo (TII)**
**What**:
- **TRI**: target register model (registers, subregs, regclasses, allocatable sets, aliasing).
- **TII**: instruction semantics (copy-like instructions, constraints, latency/cost hooks).
**Why RA cares**: Target hooks define legality and costs (e.g., which regs are reserved, copy costs, which classes are compatible).

### 3) **VirtRegMap (VRM)**
**What**: Final mapping from (possibly split) vregs → physregs or stack slots.  
**Why RA cares**: This is effectively the allocator’s “answer sheet” and is used to rewrite MIR.

---

## B) Registers and constraints

### 4) **vreg (virtual register)**
**What**: Compiler-created register, shown in MIR as `%0`, `%42`.  
**Why RA cares**: RA assigns it a physreg or spills it.

```text
%3:gr64 = COPY $rdi
```

### 5) **physreg (physical register)**
**What**: Real hardware register.
- X86: `$rax`, `$xmm0`
- AMDGPU: SGPR/VGPR physical regs + special regs (`EXEC`, `VCC`, `SCC`, `M0`)
- NVPTX: PTX regs later mapped to hardware; still modeled as machine registers in backend

**Why RA cares**: The final “colors”.

```text
$rax = COPY %3
```

### 6) **RegClass (Register Class)**
**What**: A set of legal physregs for a value (X86 `GR64`, `XMM`; AMDGPU `SReg_32`, `VReg_32`).  
**Why RA cares**: Candidate pool + legality of coalescing/assignment.

```text
%7:gr8 = COPY %8     ; X86: must be an 8-bit-capable GPR
%9:sreg_32 = ...     ; AMDGPU scalar register class
%10:vreg_32 = ...    ; AMDGPU vector register class
```

### 7) **RegBank (Register Bank)** (GlobalISel)
**What**: Coarse grouping of registers by kind (e.g., GPR vs FPR; AMDGPU SGPR-like vs VGPR-like separation).  
**Diff vs RegClass**:
- **RegBank**: chosen during GlobalISel to make instruction selection legal.
- **RegClass**: final concrete allocation constraint for RA.

**Why RA cares (indirectly)**: Bank choice strongly influences the final regclass, which drives pressure and spill behavior.

**Conceptual flow:**
```text
; Generic GlobalISel:
%1(s64) = G_ADD %a, %b

; After RegBankSelect:
%1 is assigned to GPR bank (vs FPR bank)

; After InstructionSelect:
%1 becomes a target vreg with a specific RegClass (e.g., X86 GR64 / AMDGPU SReg_64 or VReg_64)
```

### 8) **Reserved / fixed registers**
**What**: Registers not allocatable (stack pointer, program counter, special regs, etc.).  
**Why RA cares**: Removed from candidate sets (TRI decides this).

---

## C) Aliasing and subregisters (huge on X86)

### 9) **Subregister / Superregister**
**What**: Overlapping registers (X86 `AL` ⊂ `AX` ⊂ `EAX` ⊂ `RAX`).  
**Why RA cares**: Overlap means interference even if names differ.

```text
; If AL is live, RA cannot allocate another overlapping live value to RAX/EAX/AX.
```

### 10) **SubRegIdx (subregister index)**
**What**: Identifies which part/lane is referenced (e.g., `sub_32bit`).  
**Why RA cares**: Partial liveness + correct interference.

```text
%0:gr64 = ...
%1:gr32 = EXTRACT_SUBREG %0, sub_32bit
```

### 11) **RegUnit**
**What**: Smallest “resource unit” used by LLVM to model aliasing/overlap.  
**Why RA cares**: Interference checks are fundamentally based on regunits.

---

## D) MIR ops that create/shape register lifetimes (less-known but common)

### 12) **COPY**
**What**: Move between registers.  
**Why RA cares**: Coalescing tries to remove these.

```text
%b:gr64 = COPY %a
```

### 13) **REG_SEQUENCE** (less-known)
**What**: Assemble a larger register from pieces plus lane indices.  
**Why RA cares**: Links lanes/subregs; can increase pressure if it forces long-lived aggregates.

```text
%lo:gr32 = ...
%hi:gr32 = ...
%pair:gr64 = REG_SEQUENCE %lo, sub_32bit, %hi, sub_32bit_hi
```

### 14) **INSERT_SUBREG / EXTRACT_SUBREG**
**What**: Insert/extract lanes or partial registers.  
**Why RA cares**: Partial liveness and extra copies if coalescing is blocked.

```text
%lane:gr32 = EXTRACT_SUBREG %vec, sub_32bit
%vec2:vr128 = INSERT_SUBREG %vec, %lane, sub_32bit
```

### 15) **SUBREG_TO_REG**
**What**: Promote a subreg value into a full register value (upper bits undef/zero per semantics).  
**Why RA cares**: Correct lane semantics and avoids false dependencies.

```text
%full:gr64 = SUBREG_TO_REG 0, %small, sub_32bit
```

### 16) **IMPLICIT_DEF**
**What**: Defines an undefined value.  
**Why RA cares**: Models “undef lanes”; can shorten true liveness.

```text
%tmp:vr128 = IMPLICIT_DEF
```

---

## E) CFG/SSA-related vocabulary that creates copies (very relevant to RA)

### 17) **Machine PHI**
**What**: SSA merge at Machine level.  
**Why RA cares**: PHIs must be lowered to copies (“phi elimination”), creating COPY chains.

**Typical shape:**
```text
bb.0:
  %v0:gr64 = ...
  br bb.2

bb.1:
  %v1:gr64 = ...
  br bb.2

bb.2:
  %v2:gr64 = PHI %v0, %bb.0, %v1, %bb.1
```

**After phi elimination (conceptual):**
```text
; copies inserted on edges into bb.2 so bb.2 can just use one incoming value
; (coalescer then tries to remove these)
```

### 18) **Two-address instruction constraint (“two-addr”)**
**What**: Some targets require output to reuse an input register (common on X86 and others).  
**Why RA cares**: Forces ties between operands; may insert copies if the tied operands aren’t already the same.

**Conceptual X86-ish idea:**
```text
; Want: dst = dst + src (dst is both input and output)
; If IR has: %out = ADD %a, %b
; Two-addr lowering may require: COPY %a -> %out, then ADD %out, %b
```

This is a major source of “why are there so many COPYs?” in pre-RA MIR.

---

## F) Calls and clobbers

### 19) **RegMask**
**What**: Call/inline-asm clobber mask describing which physregs are overwritten.  
**Why RA cares**: Live-across-call values can’t be placed in clobbered registers.

```text
CALL64pcrel32 @foo, regmask(...)
; If %x is live across CALL, it must avoid caller-clobbered regs (or be spilled).
```

### 20) Caller-saved / callee-saved
**What**: ABI preservation sets.  
**Why RA cares**: Values live across calls often need callee-saved regs or spills.

---

## G) Liveness / interference model (Greedy RA backbone)

### 21) **SlotIndex**
**What**: Linear positions around instructions (before/after) used for liveness boundaries.  
**Why RA cares**: Precise overlap tests and spill insertion points.

### 22) **LiveRange / LiveInterval**
**What**: Where a value is live (segments); LiveInterval is the main container for a vreg.  
**Why RA cares**: Allocation is done over LiveIntervals (especially in Greedy RA).

### 23) **VNInfo (value number)**
**What**: Tracks distinct reaching definitions inside a LiveInterval.  
**Why RA cares**: Needed for correct splitting and rewriting.

### 24) **Interference**
**What**: Two values can’t share the same physreg (or overlapping regunits) at the same time.  
**Why RA cares**: Determines if a candidate assignment is legal.

### 25) **LiveRegMatrix**
**What**: Tracks which physregs are occupied over which liveness segments.  
**Why RA cares**: Fast “does this candidate interfere?” checks; supports eviction.

### 26) **Register pressure / pressure sets**
**What**: How many regs of certain kinds are simultaneously live (often tracked per class/set).  
**Why RA cares**: Guides heuristics (splitting, spilling, coalescing decisions).

### 27) **Block frequency / Branch probability (BFI/BPI)**
**What**: Profile-guided estimates of execution frequency.  
**Why RA cares**: Feeds **spill weight**: spilling in hot blocks is expensive.

---

## H) Allocation actions (what the allocator *does*)

### 28) **Coalescing**
**What**: Remove a COPY by assigning both ends the same physreg / merging intervals.  
**Why RA cares**: Fewer moves, but can increase live range length (raising pressure).

```text
%a = ...
%b = COPY %a
; coalesce => delete COPY if legal/beneficial
```

### 29) **Hint (allocation hint)**
**What**: Preference toward a particular physreg or coalescing-friendly placement.  
**Why RA cares**: Reduces moves, helps satisfy fixed-reg constraints.

### 30) **Eviction**
**What**: Kick another interval out of a physreg to allocate a higher priority one.  
**Why RA cares**: Avoids spills by reshuffling assignments.

### 31) **Splitting**
**What**: Break one interval into smaller intervals (keep hot parts in regs; spill cold parts).  
**Why RA cares**: Primary way Greedy RA reduces spill cost and fits under pressure.

```text
; %vreg  ->  %vreg.split0 (kept in reg) + %vreg.split1 (spilled)
; boundary COPYs inserted
```

### 32) **Spill / Reload**
**What**: Store to stack slot / load back.  
**Why RA cares**: Expensive; minimized especially in loops.

### 33) **Spill slot / stack slot (MachineFrameInfo)**
**What**: Stack object used to hold spilled values.  
**Why RA cares**: Spills are materialized as loads/stores to stack objects.

### 34) **FrameIndex (FI)** (less-known but essential for reading MIR)
**What**: Symbolic reference to a stack slot before final frame layout.  
**Why RA cares**: Spills/reloads in MIR often reference `fi#N` until PEI resolves it.

**Typical MIR spill/reload shape (conceptual):**
```text
%spilltmp:gr64 = COPY %v

MOV64mr $rsp, 1, $noreg, fi#3, $noreg, %spilltmp   ; store to frame index 3
%v2:gr64 = MOV64rm $rsp, 1, $noreg, fi#3, $noreg   ; load from frame index 3
```

### 35) **Rematerialization (remat)**
**What**: Recompute a value instead of reloading from memory (e.g., re-emit an immediate move).  
**Why RA cares**: Can be cheaper than reload.

---

## I) Allocator variants and supporting components

### 36) **RegAllocGreedy**
**What**: Default production allocator for many targets; global-ish via LiveIntervals, eviction, splitting.  
**Why RA cares**: Most RA debugging vocabulary matches this allocator.

### 37) **RegAllocFast**
**What**: Simpler/faster allocator (often used at `-O0`).  
**Why RA cares**: Different behavior: more local decisions, typically less splitting sophistication.

### 38) **Spiller / InlineSpiller**
**What**: Inserts spills/reloads and rewrites affected instructions.  
**Why RA cares**: Implements spill decisions; interacts with remat.

### 39) **LiveRangeEdit / SplitKit**
**What**: Utilities used to implement splitting safely and update liveness.  
**Why RA cares**: You’ll see these names when reading splitting code.

### 40) **Register scavenger**
**What**: Late mechanism to find a temporary physreg when needed after RA (or spill something as a last resort).  
**Why RA cares**: Explains “mysterious late spills/copies” appearing after RA.

### 41) **Post-RA scheduling / MachineScheduler**
**What**: Scheduling after RA must respect physreg assignments and hazards.  
**Why RA cares**: Can change lifetimes slightly and must obey fixed regs; sometimes triggers extra moves/scavenging.

---

## J) Target-specific “extra vocabulary” 
### X86 highlights
- **Aliasing** dominates: `AL`/`AX`/`EAX`/`RAX` overlap → regunit interference.
- **GR8** and (on some modes) high-8 regs (`AH/BH/CH/DH`) quirks can restrict legal allocations.
- **Two-address** patterns and fixed-reg constraints generate COPYs and hints.

### AMDGPU
- **SGPR vs VGPR**: hard separation; many instructions require one kind.
- **Special regs**:
  - `EXEC`: lane mask / execution mask
  - `VCC`: vector condition codes / carry
  - `SCC`: scalar condition code
  - `M0`: special scalar register used for certain ops
- **Occupancy**:
  - Register usage (VGPR/SGPR count) impacts waves resident → performance.
  - RA quality can mean “fewer registers” even if it costs a few extra instructions.

### NVPTX 
- **Predicate registers** (predication is common in PTX ISA).
- **Occupancy sensitivity**: registers per thread constrain active warps.
- **Local memory spills**: spills typically go to local memory (slow vs registers), so pressure matters.

---

## Belongs to
- **GlobalISel constraints**: `RegBank`, `RegisterBankInfo`
- **Register constraints**: RegClass, subregs, regunits, reserved regs
- **CFG/SSA lowering**: PHI, phi elimination, two-address
- **Liveness core**: SlotIndex, LiveIntervals, VNInfo, interference, LiveRegMatrix
- **Allocator actions**: coalescing, hints, eviction, splitting, spills/reloads, remat
- **Where spills go**: MFI stack slots, FrameIndex (`fi#N`)
- **Late surprises**: register scavenger, post-RA scheduling
