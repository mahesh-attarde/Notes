KnownBits (often called **known-bits analysis**) 
- dataflow analysis used in LLVM to track, for each integer value, which bits are **definitely 0**, **definitely 1**, and which are **unknown** at a given program point.
- It’s a cornerstone for many optimizations because it lets LLVM prove facts like “this value is non-negative”, “these low bits are zero (alignment)”, “this mask is redundant”, etc.

## The core idea
For an N-bit integer value `V`, the analysis computes two N-bit masks:

- **KnownZero**: bit *i* = 1 means `V[i]` is definitely 0  
- **KnownOne**: bit *i* = 1 means `V[i]` is definitely 1  

Bits that are 0 in both masks are **unknown** (could be 0 or 1). A well-formed result never has a bit set in both `KnownZero` and `KnownOne`.

You can think of it as representing a set of possible values. Example for 8-bit values:

- `KnownOne = 0b00010000`
- `KnownZero = 0b00001111`

Then bits 0–3 are 0, bit 4 is 1, bits 5–7 unknown, so `V` must be one of:

- `0b00010000` (16), `0b00110000` (48), `0b01010000` (80), `0b01110000` (112), ... up to `0b11110000` (240)

## Why it’s useful

KnownBits enables or strengthens many transforms, for example:

- remove redundant masks: `(x & 255)` might be removable if high bits already known zero
- infer alignment: if low k bits are known zero, the value is a multiple of `2^k`
- prove non-negativity / sign facts using known sign bit
- simplify comparisons / branches using bit facts
- detect overflow impossibility in some cases

## Examples

### 1) AND with a constant mask
```c
uint8_t y = x & 0xF0;   // 11110000
```

Resulting known bits for `y`:
- low 4 bits are **definitely 0**
  - `KnownZero` has bits 0–3 set
- high 4 bits depend on `x`
  - unknown unless `x` had more info

So:
- `KnownZero = 0b00001111`
- `KnownOne  = (KnownOne(x) & 0b11110000)` (whatever ones `x` already guaranteed in those positions)

Optimization enabled: later code like `(y & 0x0F)` is provably always 0.

---

### 2) OR with a constant sets known ones
```c
uint32_t y = x | 0x3;   // ...0011
```

Bits 0 and 1 become **definitely 1** regardless of `x`:
- `KnownOne(y)` includes bits 0–1
- Higher bits follow from `x`

So:
- `KnownOne = KnownOne(x) | 0x3`
- `KnownZero = KnownZero(x) & ~0x3` (because those bits are no longer zero)

This can prove things like `y` is odd (`bit0` known 1).

---

### 3) Shift-left introduces known zeros
```c
uint32_t y = x << 3;
```

The bottom 3 bits of `y` are always zero:
- `KnownZero(y)` includes bits 0,1,2 set
- Known bits from `x` shift upward

This is frequently used to infer **alignment**:
- if `(ptrtoint p) << 3` then it’s a multiple of 8 in integer space (low 3 bits zero)

---

### 4) Add with a small constant can create partial knowledge (sometimes)
```c
uint8_t y = (x & 0xF0) + 1;
```

We know `x & 0xF0` ends in `0000`. Adding 1 forces the low nibble to become `0001` **without carry into bit 4** (since low nibble was 0).

So for `y`:
- low 4 bits become **known**: `0001`
  - bit 0 known 1
  - bits 1–3 known 0
- higher bits are the same as `(x & 0xF0)` (unknown high nibble)

Thus:
- `KnownOne` includes `0b00000001`
- `KnownZero` includes `0b00001110`

This can enable simplifications like `(y & 0x0F) == 1` always true.

---

### 5) Proving non-negativity via known sign bit
For signed 32-bit `int32_t` values, the sign bit is bit 31.

If KnownBits proves:
- bit 31 is **known 0** (`KnownZero` has bit31 set)

Then `x >= 0` is always true (no negative values possible). This can simplify comparisons and remove branches.

Conversely, if bit 31 is **known 1**, then `x < 0` is always true.


### 6) Detecting “power of two” / single-bit patterns (limited but common)
If KnownBits shows that:
- exactly one bit may be 1, and all others are known 0, then the value is a power of two.
More commonly, KnownBits is used as an ingredient with other reasoning (like demanded bits / range analysis) to prove such properties.

## How LLVM uses 
KnownBits is typically computed by “walking” the defining operation and combining info from operands, using rules per opcode (AND/OR/XOR/shift/add/sub/zext/sext/trunc, etc.), plus facts from:
- constants
- comparisons and control flow (sometimes via other analyses)
- assumptions (`llvm.assume`)
- value tracking utilities (e.g., computeKnownBits style queries)
