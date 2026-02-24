# Notes for working towards Legality and Combiner
### LLVM GlobalISel Legality Rules
+ FLOW
1. Computes a `LegalityQuery` (opcode + a list of **types at certain indices**).
2. Finds the first rule that matches.
3. Returns an **action**:
   - **Legal** (do nothing)
   - **Widen/Narrow/Clamp scalar**
   - **More/Fewer elements** (vector length changes)
   - **SplitVector / Scalarize**
   - **Lower / Custom / Libcall**
Then it rewrites the instruction until everything becomes legal.


+ Type indices: "Type n"  meaning
Legality rules does not use terms “dst” and “src” directly, uses **type indices**.
- **Type index 0** is often the *result type* (for many instr).
- Type index 1 is often the *first operand type* that matters (for some ops it’s the same as 0 and may not be tracked separately).
- For ops where operand types differ (extend, trunc, shifts, loads), multiple indices matter.

+ why multiple indices exist
For `G_ZEXT dst, src`:
- `dst` is wider than `src`, so legality depends on the **pair** `(dstType, srcType)`.
For `G_SHL dst, val, amt`:
- legality depends on **value type** and **shift-amount type** together.

+ set matching vs tuple matching
 - Single-index set matching
	```cpp
	.legalFor({s16, s32})
	```
	Means: instruction is legal if **type[0]** is `s16` OR `s32`.
	This is a simple membership test `type[0] ∈ {s16, s32}`

  - Multi-index tuple matching (the important bit)
	```cpp
	.legalFor({{s16, s32}, {s32, s64}})
	```
	This is a 2-index overload. It means:
		- Look at `(type[0], type[1])` as a *pair*
		- That pair must match one of the allowed tuples exactly.
	Allowed:
		- `(s16, s32)` or `(s32, s64)`
	Not allowed:
		- `(s16, s64)` even though both `s16` and `s64` appear somewhere.
	Formally:
		- `(type[0], type[1]) ∈ {(s16,s32), (s32,s64)}`

    This is used to express **relationships** between operand/result types.

+ `legalFor(...)` — declaring “native” legal types

 - Scalar-only examples (X86-flavored)
   X86 supports for `s32` and `s64` integer ALU ops.

- `G_ADD` legal for `s32` and `s64`.
   ```cpp
   getActionDefinitionsBuilder(G_ADD)
     .legalFor({s32, s64});
   ```
  vector adds are legal for `v4s32` (128-bit) and `v8s32` (256-bit)
   ```cpp
   getActionDefinitionsBuilder(G_ADD)
     .legalFor({v4s32, v8s32});
   ```

+ Tuple `legalFor(...)`: extend/trunc patterns
- Allow `ZEXT s16->s32` and `s32->s64`, but not `s16->s64` directly.
  ```cpp
  getActionDefinitionsBuilder(G_ZEXT)
    .legalFor({{s32, s16}, {s64, s32}});
  ```

+ Scalar size-changing actions
- act on **scalar bitwidth** at a given type index.
- `widenScalarFor(...)` if the scalar type at some index is in this set, legalization uses **WidenScalar**.
- If you see `G_ADD s8` or `s16`, widen to `s32` because 32-bit ops are the “comfortable” baseline.

```cpp
getActionDefinitionsBuilder(G_ADD)
  .legalFor({s32, s64})
  .widenScalarFor({s8, s16});
```
- Result behavior (conceptually):
    G_ADD s8` becomes:
    - `zext`/`sext` operands to `s32`
    - `G_ADD s32`
    - `trunc` back to `s8` if needed
- `narrowScalarFor(...)`
	- If the IR produced `s128` arithmetic (common with i128), but target only wants up to `s64`, narrow or split/lower.

+ `clampScalar(...)` force scalar size into a range (widen if too small, narrow if too big).
	- Only want `s32` and `s64` sizes, so clamp index 0 into `[32..64]`.

    ```cpp
    getActionDefinitionsBuilder(G_XOR)
      .legalFor({s32, s64})
      .clampScalar(0, s32, s64);
    ```
	Behavior:
		- `s8` -> widened up (typically to `s32`)
		- `s128` -> narrowed down (typically to `s64`)


+ Vector element-count actions (More/Fewer elements)
- These change **number of lanes**, not lane type.
- `moreElementsToNextPow2(...)`
	Useful when hardware prefers powers of two lanes.
	Example:
	- `v3s32` -> `v4s32`
	- `v5s32` -> `v8s32`

    ```cpp
    getActionDefinitionsBuilder(G_ADD)
      .legalFor({v4s32, v8s32})
      .moreElementsToNextPow2(0);
    ```

- `fewerElementsToMin(...)`
	Reduce lane count until it’s <= some minimum supported.
	
	Example:
	- `v16s32` -> `v8s32` -> `v4s32`

    ```cpp
    getActionDefinitionsBuilder(G_ADD)
      .legalFor({v4s32, v8s32})
      .fewerElementsToMin(0, 4);
    ```

- `clampNumElements(...)` Clamp lanes into a range.
	- Only support between 4 and 8 lanes.
    
    ```cpp
    getActionDefinitionsBuilder(G_ADD)
      .clampNumElements(0, 4, 8);
    ```

+ Vector fallback actions: `splitVector` vs `scalarize`

- `splitVector(...)` Split a vector into multiple smaller vectors.
	- If `v8s32` (256-bit) is not supported but `v4s32` is, split into two `v4s32`.

- `scalarize(...)` Break vector op into scalar ops per lane.
  This is a last resort and tends to be slower.
- if vector op isn’t supported, scalarize lane-by-lane.

+  Multi-index legality and actions (X86-style patterns)

This is where tuple matching is essential.

* Shifts: value type + amount type
On many targets, the **shift amount** is treated as `s8`/`s32` in different lowering phases, but a target may want to legalize it to a single type (often `s32` in GISel legality).

Example intent:
- allow `(value, amt)` to be:
  - `(s32, s32)` or `(s64, s32)`

  ```cpp
  getActionDefinitionsBuilder(G_SHL)
    .legalFor({{s32, s32}, {s64, s32}});
  ```
- `shl s64, s64` is **not** legal unless you explicitly add `{s64, s64}`
- This is exactly the “no mixing across tuples” behavior.

* Extend and trunc: enforcing legal pairs
Example intent:
- Trunc legal only for `s64->s32` and `s32->s16`

  ```cpp
  getActionDefinitionsBuilder(G_EXT)
    .legalFor({{s32, s64}, {s16, s32}});
  ```
* Loads: result type + pointer type/address space
- legality often depends on pointer type and address space.
- legal loads:
  - `load s32 from p0`
  - `load s64 from p0`

```cpp
getActionDefinitionsBuilder(G_LOAD)
  .legalFor({{s32, p0}, {s64, p0}});
```

+ When types aren’t fixable by widening/narrowing: `lower`, `custom`, `libcall`

* `lower(...)` Use when an op should be expanded to simpler/legal ops.
	- `G_CTLZ` might be expanded using compares and bit tricks if no direct instruction.

    ```cpp
    getActionDefinitionsBuilder(G_CTLZ)
      .lower();
    ```

* `custom(...)` Use when the target will implement a special legalization routine.
	- X86 has many special cases depending on SSE/AVX/AVX512, and may want custom legalization for some vector ops.

   ```cpp
   getActionDefinitionsBuilder(G_FNEG)
     .custom();
   ```

* `libcall(...)` Use when it’s best/necessary to call a runtime helper (often for FP math).


## Combiner TODO

## GISEL Matcher Table  Generation
