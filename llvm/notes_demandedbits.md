# Demanded Bits Analysis:

**DemandedBits** analysis is a backward, bit-level liveness analysis over LLVM IR integer values. It answers:

> For a given integer-producing instruction or use, which result bits can affect observable program behavior?

A bit is **demanded** if changing that bit may change control flow, memory side effects, externally visible values, or some live downstream computation. 
A bit is **not demanded** if it may be replaced by either `0` or `1` without affecting relevant program behavior.

LLVM documents the core idea directly in `DemandedBits.h` and `DemandedBits.cpp`:

- Header https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Analysis/DemandedBits.h#L8-L17
- Implementation  https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L9-L17

Canonical example:

```llvm
%1 = add i32 %x, %y
%2 = trunc i32 %1 to i16
```

Only bits `[0..15]` of `%1` are demanded because `%2` discards the high 16 bits.

DemandedBits is mainly consumed by:

1. **BDCE** — Bit-Tracking Dead Code Elimination.
2. **Loop vectorization / vector utilities** — computing smaller legal integer widths.
3. **Reduction analysis** — choosing narrower recurrence types.
4. Related, but separate, demanded-bit simplification machinery in InstCombine and SelectionDAG.


## 2. Main API

Primary declaration:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Analysis/DemandedBits.h#L41-L115

Key class:

```cpp
class DemandedBits {
public:
  DemandedBits(Function &F, AssumptionCache &AC, DominatorTree &DT);

  APInt getDemandedBits(Instruction *I);
  APInt getDemandedBits(Use *U);

  bool isInstructionDead(Instruction *I);
  bool isUseDead(Use *U);

  void print(raw_ostream &OS);

  static APInt determineLiveOperandBitsAdd(...);
  static APInt determineLiveOperandBitsSub(...);
};
```

Important internal state:

- `DenseMap<Instruction *, APInt> AliveBits`
  - Maps an instruction to the demanded-bit mask of its result.
- `SmallPtrSet<Use *, 16> DeadUses`
  - Tracks operand uses with no demanded bits.
- `SmallPtrSet<Instruction *, 32> Visited`
  - Tracks visited non-integer instructions.
- `AssumptionCache` and `DominatorTree`
  - Used when querying `computeKnownBits`.

Analysis registration:

- Implementation: https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L639-L652
- New PM registry: https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Passes/PassRegistry.def#L350-L368


```bash
opt -passes='print<demanded-bits>' input.ll -disable-output
```

## 3. Conceptual model

### 3.1 Bit liveness lattice

For an `n`-bit integer value, DemandBits represents liveness as an `APInt` mask:

- Bit `1`: demanded / alive.
- Bit `0`: not demanded / dead.

For example, for an `i32` value:

```text
0x0000ffff
```

means only low 16 bits are demanded.

The analysis is monotonic and backward-propagating:

```text
uses / roots
   ↓ backward transfer
operands
   ↓ backward transfer
operand operands
```

At joins, masks are unioned:

```text
AliveBits[V] = AliveBits[V] | newly_required_bits
```

A fixed point is reached when no instruction gains new demanded bits.

### 3.2 Roots

Demand starts from “always live” instructions:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L48-L50

```cpp
static bool isAlwaysLive(Instruction *I) {
  return I->isTerminator() || I->isEHPad() || I->mayHaveSideEffects();
}
```

So these are roots:

- terminators,
- EH pads,
- side-effecting instructions.

For integer-typed root instructions, the initial alive mask is empty, and operands are then processed according to instruction semantics. For non-integer roots, integer operands are conservatively marked fully demanded.

Root setup:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L354-L399


## 4. Algorithm

Main fixed-point engine:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L354-L465

High-level pseudocode:

```text
performAnalysis():
  if already analyzed:
    return

  clear Visited, AliveBits, DeadUses

  Worklist = {}

  for each instruction I in function:
    if I is always-live:
      if I returns integer:
        AliveBits[I] = 0
        enqueue I
      else:
        for each operand O:
          if O is integer instruction:
            AliveBits[O] = all ones
          enqueue operand instruction

  while Worklist not empty:
    UserI = pop Worklist

    if UserI returns integer:
      AOut = AliveBits[UserI]
      InputIsKnownDead = AOut == 0 && !isAlwaysLive(UserI)

    for each operand use OI of UserI:
      if OI is integer:
        if InputIsKnownDead:
          AB = 0
        else:
          AB = transfer(UserI, operand number, AOut)

        if AB == 0:
          DeadUses.insert(OI)

        if OI is instruction:
          Old = AliveBits[OI]
          New = Old | AB
          if New changed:
            AliveBits[OI] = New
            enqueue OI
      else if OI is instruction and not visited:
        enqueue OI
```

The key transfer function is:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L52-L351

```cpp
void DemandedBits::determineLiveOperandBits(...)
```


## 5. Transfer-function intuition

DemandedBits propagates demanded result bits backward to demanded operand bits according to opcode semantics.

### 5.1 Casts

Source:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L322-L337

Rules:

| Instruction | Operand demand |
|---|---|
| `trunc` | zero-extend output demand to source width |
| `zext` | truncate output demand to source width |
| `sext` | truncate output demand, plus demand source sign bit if any extended sign bits are demanded |

Example:

```llvm
%a = sext i8 %x to i32
%b = trunc i32 %a to i8
```

The high 24 bits of `%a` are not demanded by `%b`; BDCE can often convert `%a` from `sext` to `zext`.

BDCE implements exactly this use case:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp#L116-L133

### 5.2 Shifts

Source:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L192-L290

Rules:

- `shl x, C`
  - demanded bits of `x` are `AOut >> C`.
  - If `nuw`/`nsw` is present, some high input bits remain demanded because poison-generating no-wrap promises must remain valid.
- `lshr x, C`
  - demanded bits of `x` are `AOut << C`.
  - If `exact`, low shifted-out bits are demanded because they must be zero.
- `ashr x, C`
  - similar to `lshr`, but the sign bit may be demanded because it is replicated.

Variable shifts use known-bit-derived min/max shift amounts:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L79-L98

### 5.3 Bitwise operations

Source:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L292-L320

Rules:

- `xor`: each demanded output bit demands corresponding input bits.
- `and`: if the other operand is known zero at a bit, this operand’s bit is dead.
- `or`: if the other operand is known one at a bit, this operand’s bit is dead.
- `phi`: demanded output bits propagate directly to operands.

Example:

```llvm
%x = and i32 %a, 255
%y = trunc i32 %x to i8
```

Only low 8 bits of `%x` are demanded. The mask `255` does not change demanded bits beyond those already kept.

BDCE simplification:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp#L136-L166

### 5.4 Add/sub

Source:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L170-L185

For simple low-bit masks, add/sub only demand the same low bits:

```cpp
if (AOut.isMask())
  AB = AOut;
```

For non-contiguous demanded bits, LLVM uses known-bits-aware carry propagation:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L559-L637

Core insight:

- A high demanded output bit may depend on lower input bits through carries.
- But known zeros/ones can stop carry propagation.
- Therefore `computeKnownBits` can reduce operand demand.

The add/sub propagators are unit-tested exhaustively for 4-bit values:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/unittests/IR/DemandedBitsTest.cpp#L1-L68

### 5.5 Mul

Source:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L186-L191

Rule:

```cpp
AB = APInt::getLowBitsSet(BitWidth, AOut.getActiveBits());
```

If only low `k` result bits are demanded, only low `k` input bits are needed. This follows from integer multiplication modulo `2^n`: low result bits do not depend on higher input bits.

### 5.6 Intrinsics

Source:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L102-L168

Handled intrinsics include:

| Intrinsic | Demand behavior |
|---|---|
| `llvm.bswap` | byte-swaps demand mask |
| `llvm.bitreverse` | bit-reverses demand mask |
| `llvm.ctlz` | demands input bits up to known possible leading-one boundary |
| `llvm.cttz` | demands input bits down to known possible trailing-one boundary |
| `llvm.fshl` / `llvm.fshr` | maps demanded output bits to funnel-shift operands |
| `llvm.{u,s}{min,max}` | if low result bits are dead, corresponding low operand bits are dead |


## 6. Interaction with KnownBits

DemandedBits frequently calls `computeKnownBits`:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L64-L78

KnownBits answers:

> Which bits of a value are provably zero or one?

DemandedBits answers:

> Which bits matter to downstream computation?

They complement each other. KnownBits improves DemandedBits precision for:

- `and`,
- `or`,
- add/sub carry propagation,
- variable shifts,
- `ctlz` / `cttz`.

Example:

```llvm
%a = and i32 %x, 255
%b = add i32 %a, 1
%c = trunc i32 %b to i8
```

DemandedBits sees only low 8 bits demanded by `trunc`. KnownBits can prove `%a` high bits are zero, preventing unnecessary demand propagation into high bits.


## 7. Query behavior

Instruction query:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L467-L476

If an instruction has an entry in `AliveBits`, return it. Otherwise return all bits demanded as a conservative fallback.

Use query:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L478-L502

For non-integer uses, all bits are treated as demanded.

Dead instruction/use queries:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L505-L533

Important semantics:

- An instruction is dead if it was not reached and is not always live.
- A use is dead if it has no demanded bits.
- Uses by always-live instructions are not considered dead.


## 8. Major use cases

## 8.1 Bit-Tracking Dead Code Elimination: `BDCE`

BDCE’s file-level comment describes the pass:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp#L9-L13

BDCE uses DemandedBits to:

1. Remove instructions with no demanded bits.
2. Replace dead operands with zero.
3. Convert `sext` to `zext` when sign-extension bits are unused.
4. Remove redundant `and` / `or` / `xor` masks.

Main implementation:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp#L96-L189

Pass entry:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp#L205-L213

Example:

```llvm
define i8 @f(i32 %x) {
entry:
  %masked = and i32 %x, 255
  %tr = trunc i32 %masked to i8
  ret i8 %tr
}
```

The `and` mask may be redundant because the `trunc` demands only low 8 bits anyway.

Another example:

```llvm
define i8 @g(i8 %x) {
entry:
  %sx = sext i8 %x to i32
  %tr = trunc i32 %sx to i8
  ret i8 %tr
}
```

The sign-extension bits are not demanded, so BDCE can replace `sext` with `zext`, and later simplifications may remove the extension entirely.

## 8.2 Loop vectorizer minimum value sizes

`computeMinimumValueSizes` uses DemandedBits to identify narrower integer widths that still preserve demanded behavior.

Declaration:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Analysis/VectorUtils.h#L319-L352

Implementation:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/VectorUtils.cpp#L793-L895

Loop vectorizer hook:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Vectorize/LoopVectorizationPlanner.cpp#L546-L550

Why this matters:

C integer promotion often turns `i8`/`i16` operations into `i32`. Scalar InstCombine may avoid shrinking them if the smaller scalar type is illegal on the target. But SIMD hardware often supports narrow vector integer types. During vectorization, knowing the demanded width lets LLVM model or generate narrower vector operations.

Example from the comment:

```llvm
%1 = load i8, ptr %p
%2 = add i8 %1, 2
%3 = load i16, ptr %q
%4 = zext i8 %2 to i32
%5 = zext i16 %3 to i32
%6 = add i32 %4, %5
%7 = trunc i32 %6 to i16
```

DemandedBits helps infer `%6` only needs 16 bits.

## 8.3 Reduction recurrence type selection

Reduction descriptors use DemandedBits to compute a smaller recurrence type.

Header comment:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Analysis/IVDescriptors.h#L175-L213

Implementation:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/IVDescriptors.cpp#L95-L124

Key logic:

```cpp
if (DB) {
  auto Mask = DB->getDemandedBits(Exit);
  MaxBitWidth = Mask.getBitWidth() - Mask.countl_zero();
}
```

If only low `k` bits of a reduction exit value are demanded, the recurrence may be computed in a smaller integer type. If demanded bits cannot reduce the width, LLVM falls back to sign-bit/value tracking.

This is used by reduction recognition:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/IVDescriptors.cpp#L1009-L1077

## 8.4 Loop vectorization legality

Loop vectorization legality stores a `DemandedBits *DB` specifically to compute minimum reduction type sizes:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Transforms/Vectorize/LoopVectorizationLegality.h#L717-L724

## 8.5 SLP vectorizer plumbing

SLPVectorizer carries a `DemandedBits *DB` analysis pointer:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Transforms/Vectorize/SLPVectorizer.h#L24-L112

This makes demanded-bit information available during SLP vectorization decisions.


## 9. Related but distinct: InstCombine demanded bits

InstCombine has its own recursive demanded-bit simplification machinery:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Transforms/InstCombine/InstCombiner.h#L469-L553

Implementation entry:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/InstCombine/InstCombineSimplifyDemanded.cpp#L125-L213

Important distinction:

- `llvm::DemandedBits` is a function analysis with a global fixed-point view.
- InstCombine’s `SimplifyDemandedBits` is a local recursive simplifier used to rewrite IR during combining.

They are conceptually related but implemented separately.


## 10. Important correctness caveats

### 10.1 Poison-generating flags matter

BDCE drops poison-generating annotations when trivializing users:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp#L73-L85

For example, `nuw`, `nsw`, and `exact` encode semantic promises. If an operand is replaced or deadened, those promises may no longer hold.

DemandedBits also preserves demand for bits needed to maintain no-wrap/exact semantics:

- `shl nuw/nsw`: https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L199-L216
- `lshr exact`: https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L227-L249
- `ashr exact`: https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L265-L288

### 10.2 Vector precision is per-bit, not per-lane

The API explicitly says vector elements are not distinguished:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Analysis/DemandedBits.h#L46-L55

For vector integer values:

> A bit is demanded if it is demanded for any vector element.

So DemandedBits cannot say “lane 0 demands bit 3 but lane 1 does not.” It collapses across lanes.

### 10.3 Non-integer values are conservative

Non-integer or pointer-like values generally return all bits demanded, or are tracked only structurally. There is a loop-vectorizer regression test ensuring `getDemandedBits()` on a pointer-typed GEP does not crash:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/test/Transforms/LoopVectorize/demanded-bits-of-pointer-instruction.ll#L1-L20

### 10.4 Analysis invalidation matters

DemandedBits depends on IR structure and use-def relationships. Many IR changes invalidate it. Example tests:

- Constraint elimination invalidation: https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/test/Transforms/ConstraintElimination/analysis-invalidation.ll#L1-L43
- Loop vectorization invalidation: https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/test/Transforms/LoopVectorize/novect-lcssa-cfg-invalidation.ll#L1-L42


## 11. Worked examples

### Example 1: trunc kills high bits

Input:

```llvm
define i16 @trunc_add(i32 %x, i32 %y) {
entry:
  %sum = add i32 %x, %y
  %t = trunc i32 %sum to i16
  ret i16 %t
}
```

Demand:

```text
%t   : low 16 bits demanded
%sum : low 16 bits demanded
%x   : low 16 bits demanded, because low 16 bits of add depend only on low 16 bits
%y   : low 16 bits demanded
```

### Example 2: sign-extension high bits unused

```llvm
define i8 @unused_sign_bits(i8 %x) {
entry:
  %sx = sext i8 %x to i32
  %t = trunc i32 %sx to i8
  ret i8 %t
}
```

DemandedBits:

```text
%sx demanded mask = 0x000000ff
```

BDCE can convert `sext` to `zext` because none of the sign-extension bits are demanded.

Relevant BDCE code:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp#L116-L133

### Example 3: redundant mask

```llvm
define i8 @redundant_and(i32 %x) {
entry:
  %a = and i32 %x, 255
  %t = trunc i32 %a to i8
  ret i8 %t
}
```

The `and` preserves exactly the low 8 bits that are demanded by the trunc. High bits are not demanded. BDCE may remove or simplify the mask.

Relevant code:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp#L136-L166

### Example 4: add carry precision

```llvm
define i32 @noncontiguous(i32 %x, i32 %y) {
entry:
  %s = add i32 %x, %y
  %m = and i32 %s, 65536
  ret i32 %m
}
```

Only bit 16 of `%s` is directly demanded, but bit 16 may depend on lower bits through carry propagation. Demand must propagate into lower bits unless KnownBits proves carry cannot cross some boundary.

Relevant add/sub carry code:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L559-L637


## 12. Testing and debugging

### Debug pass manager

```bash
opt -passes='require<demanded-bits>,bdce' \
    -debug-pass-manager \
    -disable-output test.ll
```

### Unit tests

Add/sub demanded-bit propagation is tested exhaustively here:

https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/unittests/IR/DemandedBitsTest.cpp#L1-L68


## 13. Source map

| Area | File / permalink |
|---|---|
| Public API | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Analysis/DemandedBits.h#L41-L115 |
| Main implementation | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp |
| Opcode transfer function | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L52-L351 |
| Worklist fixed point | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L354-L465 |
| Query APIs | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L467-L533 |
| Add/sub helpers | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L559-L637 |
| Analysis registration | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/DemandedBits.cpp#L639-L652 |
| Pass registry | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Passes/PassRegistry.def#L350-L368 |
| BDCE consumer | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Scalar/BDCE.cpp |
| Vector minimum widths declaration | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Analysis/VectorUtils.h#L319-L352 |
| Vector minimum widths implementation | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/VectorUtils.cpp#L793-L895 |
| Loop vectorizer minimal widths hook | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Transforms/Vectorize/LoopVectorizationPlanner.cpp#L546-L550 |
| Reduction recurrence type | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/lib/Analysis/IVDescriptors.cpp#L95-L124 |
| Reduction APIs | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/include/llvm/Analysis/IVDescriptors.h#L175-L213 |
| Unit tests | https://github.com/llvm/llvm-project/blob/dc79e2a9a1c6809c940ef87c405ed8590768cceb/llvm/unittests/IR/DemandedBitsTest.cpp#L1-L68 |


## 14. Mental model for compiler engineers

DemandedBits is best understood as:

```text
DCE lifted from value granularity to bit granularity.
```

Classic DCE asks:

> Is this instruction’s result used?

DemandedBits asks:

> Which bits of this instruction’s result are used?

This enables optimizations that are invisible to ordinary SSA-level liveness:

- eliminate computations that produce only dead high bits,
- shrink arithmetic widths,
- remove masks,
- replace dead operands,
- reduce vectorization costs,
- choose narrower reduction recurrence types.

Its precision comes from combining:

1. backward liveness,
2. opcode-specific transfer functions,
3. `KnownBits`,
4. dominance/assumption-aware value tracking.

Its conservatism comes from:

1. side effects and terminators as roots,
2. all-bits-demanded fallback for unknown/non-integer cases,
3. careful preservation of poison-generating semantics,
4. per-vector-bit rather than per-lane precision.
