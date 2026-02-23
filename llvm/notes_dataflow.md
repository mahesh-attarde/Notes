# SSA-based “dataflow” 

LLVM propagates facts along **def-use edges** (use lists) and uses CFG reasoning only where necessary (e.g., PHIs, dominance, executable edges). 
This yields **sparse** analyses: you touch only values/instructions that matter.
Instead of iterating over basic blocks and propagating sets, 

## 1) def-use chains are the dataflow graph
In LLVM IR, every SSA `Value` knows its users. Many analyses do:
1. Start with a worklist of “interesting” Values/Instructions.
2. When you learn something new about a Value, visit its users.
3. Recompute a “fact” for those users from their operands’ facts.
4. If a user’s fact changed, push it.
This is classic monotone dataflow, just on the **SSA value graph** rather than the CFG.

### Example (def-use propagation)
```llvm
define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = mul i32 %a, 8
  ret i32 %b
}
```
A sparse range/bit analysis can:
- Learn something about `%x`
- Propagate to `%a` (because `%a` uses `%x`)
- Then propagate to `%b` (because `%b` uses `%a`)
No need to visit unrelated blocks or instructions.

## 2) The hard part in SSA: merges (PHI nodes) and control-flow facts
PHIs merge values from multiple predecessors:
```llvm
merge:
  %p = phi i32 [ %v1, %pred1 ], [ %v2, %pred2 ]
```

Any SSA-based analysis must define how facts merge at PHIs:
- Constant propagation: `%p` is constant only if all incoming executable edges provide the *same* constant.
- Range/bit facts: merge with a **join** operator (often “least precise that covers all possibilities”), e.g., range union, KnownBits intersection-ish, etc.
Also, some analyses need to know **which predecessor edges are feasible** (path sensitivity). That’s where SCCP stands out.

## 3) SCCP in LLVM: sparse dataflow + executable CFG edges
**SCCP (Sparse Conditional Constant Propagation)** is the canonical LLVM SSA-based dataflow analysis/transform.
### 3.1 Lattice
Each SSA value is mapped to a small lattice, typically:

- **⊥ (Undefined / not yet known)**  
- **Constant(C)** (a specific constant)
- **Overdefined (⊤)** meaning “not a single constant” (could vary)

Meet/join rules:
- Constant(C) ⊔ Constant(C) = Constant(C)
- Constant(C1) ⊔ Constant(C2≠C1) = Overdefined
- Anything ⊔ Overdefined = Overdefined

### 3.2 Two coupled worklists
SCCP solves two problems together:
1) Dataflow over SSA values (constant-ness)
2) Reachability of blocks/edges (executability)

So it maintains:
- **Value worklist**: instructions whose output lattice might improve
- **CFG/worklist**: basic blocks or edges that become executable

### 3.3 Mechanics with a concrete example
```llvm
define i32 @g(i1 %c) {
entry:
  br i1 %c, label %T, label %F

T:
  %x = add i32 40, 2
  br label %M

F:
  %y = add i32 1, 2
  br label %M

M:
  %p = phi i32 [ %x, %T ], [ %y, %F ]
  ret i32 %p
}
```

- Initially, only `entry` is executable.
- `%c` is unknown → both edges `entry->T` and `entry->F` become executable (unless later proven constant).
- In `T`, SCCP computes `%x = 42` constant.
- In `F`, `%y = 3` constant.
- At PHI `%p`: join(Constant(42), Constant(3)) = Overdefined ⇒ `%p` not constant.

Now if `%c` were proven constant `true`, SCCP would:
- Mark only `entry->T` executable
- Ignore facts from `F`
- Then `%p` becomes Constant(42)

That “ignore infeasible predecessor” aspect is the “conditional” in SCCP.

### 3.4 What SCCP enables
Once you have constants and reachable blocks:
- Replace instructions with constants
- Fold branches with constant conditions
- Remove unreachable blocks
- Simplify PHIs (drop incoming from dead preds)

In LLVM, SCCP exists both as:
- an **analysis-like solver**
- and a **transform pass** that rewrites IR based on results.

TODO: API and Usage

## 4) Value Tracking / KnownBits: SSA facts about bits (not full constants)
A lot of LLVM’s middle-end cares about partial information:
- “bit 0 is always 0”
- “this value is non-negative”
- “these top bits are sign-extended”
- “these bits are unknown”

### 4.1 KnownBits idea
For an `n`-bit integer value `V`, represent:
- `KnownZero` bitmask
- `KnownOne` bitmask
with invariant `KnownZero & KnownOne == 0`.

Example:
- If `V` is known even: LSB is 0  
  `KnownZero` has bit0 = 1

Operations propagate these masks:
- `shl`: shifts known bits accordingly
- `and`: intersects known-one/known-zero per boolean algebra
- `or`, `xor`, `add`: more complex (may lose precision)

### 4.2 Sparse propagation
This is still sparse: to compute KnownBits for `%b` you query operands, and those queries recursively walk def-use backwards (often memoized + with depth limits / cycle handling). In practice, LLVM uses a mix of:
- query-based recursive reasoning
- caching
- and sometimes local fixpoint for cycles (PHIs).

### 4.3 Small example
```llvm
%v = and i32 %x, 254  ; 0b11111110
```
Result: bit0 is known zero (even), regardless of `%x`.
So optimizations can replace `(urem %v, 2)` with `0`, etc.

## 5) Lazy Value Information (LVI): SSA value facts *at a program point*
SCCP gives global-ish facts; KnownBits often gives “context-free” facts.

**LVI** answers: *what can we prove about value V at instruction I?*  
It is **program-point sensitive** and uses dominance, conditions, and assumptions.

### 5.1 Typical query
- “At this `load`, is `%idx` within bounds?”
- “At this `if`, can `%p` be null?”

### 5.2 How it works 
LVI tries multiple sources, usually in escalating cost:
1. **Immediate simplification**: is `V` a constant / can be folded locally?
2. **Dominating conditions**: if you’re in a block dominated by `if (V == 0)` false-edge, then inside that region you know `V != 0`.
3. **PHI reasoning**: if you’re at a join, LVI queries incoming values along feasible/dominating edges.
4. **Assumptions**: `llvm.assume` and `!range` metadata, etc.

So LVI is “SSA-based” because:
- it reasons in terms of SSA Values
- it uses def-use / PHI structure
- and uses dominance/CFG to know which predicates hold at the point.

### 5.3 Example
```llvm
br i1 %c, label %T, label %F
T:
  %p = ... ; some pointer
  br label %M
F:
  br label %M
M:
  %q = phi ptr [ %p, %T ], [ null, %F ]
  ; At this point, q may be null.
```
But if later we have:
```llvm
M:
  %isnull = icmp eq ptr %q, null
  br i1 %isnull, label %Null, label %NonNull
NonNull:
  ; inside here, LVI can prove %q != null
```
That’s a key distinction: SCCP can’t necessarily decide `%q` globally, but LVI can decide `%q` *in NonNull*.

## 6) Handling cycles in SSA: loops and PHIs
SSA value graphs can have cycles via PHIs:
```llvm
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit
```

Analyses must avoid infinite recursion / oscillation:
- SCCP uses a worklist and monotone lattice; it converges because facts only move upward (⊥ → Constant → Overdefined).
- Range/bit analyses often use **widening** or conservative joins at PHIs to ensure termination.
- Many query-based utilities impose recursion depth limits and fall back to “unknown” to stay compile-time-safe.

Also: loop reasoning is often delegated to **ScalarEvolution (SCEV)**, which is SSA-based but specialized for induction/recurrent expressions.

## 7) How these SSA-based analyses get consumed by optimizations
Common consumers:
- **InstCombine**: uses constant folding + KnownBits + demand-driven simplification
- **SimplifyCFG**: uses constant conditions / SCCP results
- **GVN / PRE / LICM**: rely on value numbering plus memory analyses; also consume constant/range facts
- **Vectorizers**: consume SCEV, LVI, known alignment/range facts

A typical pattern is: optimization queries multiple analyses, each giving partial information; if any proves something, do the rewrite.

## 8) LLVM SSA-based dataflow as 3 “styles”:

1) **Solver-based sparse propagation** (SCCP, Attributor-style):  
   explicit lattice + worklist, converges globally.

2) **Query-based backward reasoning** (ValueTracking/KnownBits):  
   ask “what do we know about V?”, recursively inspect its definition.

3) **Program-point-sensitive predicate reasoning** (LVI):  
   ask “what do we know about V *here*?”, using dominance + conditions.

They’re all SSA-centric, but with different tradeoffs between precision and compile-time.

### TODO
1) **SCCP internals**: exact lattice, how it treats PHIs/selects/calls, how it marks edges executable, and how rewriting happens.
2) **KnownBits/value tracking**: transfer functions for common ops, how PHIs merge, how LLVM avoids expensive fixpoints.
3) **LVI**: how it walks dominating conditions, what predicates it tracks (icmp, range metadata, assumes), and typical pitfalls.
4) **How SSA-based facts interact with MemorySSA/AA** (still SSA-ish, but for memory).
