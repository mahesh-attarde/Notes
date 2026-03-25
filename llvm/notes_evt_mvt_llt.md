### 1. `EVT` – Extended Value Type
**Namespace:** `llvm`  
**Header:** `llvm/CodeGen/ValueTypes.h`
`EVT` stands for **Extended Value Type**. It’s a codegen-level type abstraction that:
- Represents all the “normal” IR scalar/vector types (`i1`, `i8`, `i32`, `float`, `<4 x i32>`, etc.), **plus** some extra types that don’t exist at the IR level.
- Can encode:
  - Integer types of arbitrary bit width, not just powers of 2.
  - Non-standard vector types.
  - Target-specific or backend-specific value types.

Typical use: in SelectionDAG-based backends and type-legalization, pattern matching, and instruction selection, when you need a “rich” notion of type that can go beyond the canonical `MVT`.

### 2. `MVT` – Machine Value Type

**Namespace:** `llvm`  
**Header:** `llvm/CodeGen/ValueTypes.h`

`MVT` stands for **Machine Value Type**. Conceptually:

- It’s a *limited* enumeration of all “legal” or canonical **machine value types** used during codegen.
- It’s generally an enum-like class (a thin wrapper around an enum) with cases like:
  - `MVT::i1`, `MVT::i8`, `MVT::i16`, `MVT::i32`, `MVT::i64`, `MVT::i128`
  - `MVT::f32`, `MVT::f64`, …
  - `MVT::v4i32`, `MVT::v8i16`, `MVT::v2f64`, …
  - `MVT::Other`, `MVT::iPTR`, etc.

Main points:

- Every `MVT` is representable as an `EVT`, but not every `EVT` has a corresponding `MVT`.
- After type legalization, operands are generally in `MVT`s that the target can handle directly.
- You often see conversions:

  ```cpp
  EVT E = ...;
  if (E.isSimple()) {
    MVT M = E.getSimpleVT();   // convert EVT → MVT for simple (enum) types
  }
  ```

### 3. `LLT` – Low-Level Type

**Namespace:** `llvm`  
**Header:** `llvm/CodeGen/GlobalISel/Types.h` (or nearby in GlobalISel)

`LLT` stands for **Low-Level Type** and is primarily used in the **GlobalISel** pipeline (the newer instruction selector).

It plays a similar conceptual role to `EVT`/`MVT`, but:

- Designed for GlobalISel instead of SelectionDAG.
- Represents:
  - Scalar integer and floating types (`LLT::scalar(32)`, etc.).
  - Vector types (`LLT::vector(NumElements, ElementSizeBits)`).
  - Pointer types (`LLT::pointer(AddressSpace, SizeInBits)`).
- It’s more *structural* (described by bit sizes, element counts, and address spaces) rather than an enum.

Example:

```cpp
LLT Ty = MRI.getType(Reg);
if (Ty.isScalar() && Ty.getSizeInBits() <= 32) { ... }
if (Ty.isPointer()) { ... }
if (Ty.isVector()) { ... }
```

`LLT` tries to be more general and less enum-bound than `MVT`, which gives GlobalISel more flexibility for arbitrary integer widths, vectors, and target-specific behavior.

## Relationship between them

- **IR-level types:** `llvm::Type` (e.g., `IntegerType::get(Context, 37)`, `VectorType::get(...)`).
- **SelectionDAG / classic ISel:**
  - `EVT` is the “extended” type used everywhere in DAG nodes.
  - `MVT` is the canonical machine type enum for legalized types.
- **GlobalISel:**
  - `LLT` is the corresponding low-level type abstraction used in `MachineIR`.

Rough mapping:

- `Type` (IR) → `EVT` (DAG/CodeGen) → `MVT` (legalized machine types)
- `Type` (IR) → `LLT` (GlobalISel)


## Other type-related classes in the backend

### 4. `Type` hierarchy (IR level, but used in backend too)

- `llvm::Type`, `IntegerType`, `PointerType`, `StructType`, `VectorType`, `ArrayType`.
- While they’re IR types, many backend components inspect them when mapping from IR to low-level types (e.g., determining calling convention, ABI lowering, etc.).

### 5. `DataLayout`

**Header:** `llvm/IR/DataLayout.h`

Represents target-specific size, alignment, and layout information:

- Maps IR `Type` → size in bits/bytes, ABI alignment, pointer sizes, address spaces, etc.
- Crucial during lowering, frame layout, and legalization.

```cpp
const DataLayout &DL = MF.getDataLayout();
uint64_t SizeInBits = DL.getTypeSizeInBits(Ty);
unsigned Align = DL.getABITypeAlignment(Ty);
```

### 6. `ISD::ArgFlagsTy` / `CCValAssign` / `CCState`

Used in **calling-convention / argument lowering**:

- `ISD::ArgFlagsTy`: per-argument flags (e.g., sign-extend, zero-extend, byval, sret).
- `CCValAssign`: describes how an argument or return value is assigned (in a register, on stack, with what type).
- `CCState`: tracks call-lowering state, including type information, registers used, and stack offsets.

These are more “type + location” descriptors than pure types, but they’re critical for backend type handling at ABI boundaries.

### 7. `LegalizerInfo`, `LegalizeActions`, `LegalizerHelper` (GlobalISel)

While not types themselves, they are heavily type-driven:

- Define which `LLT`s are *legal*, and what to do with illegal ones (narrow, widen, scalarize, etc.).
- They interact with `LLT` to decide transformations.

### 8. `MachineMemOperand::Flags`, `MemoryLocation`, `MachinePointerInfo`

These encode memory-type-related info:

- Size, alignment, volatility, atomic ordering, etc.
- Work together with `LLT`/`MVT` and `DataLayout` to capture the “type” of memory accesses.

### 9. Target-specific type helpers

Many targets add their own layer of type abstraction or queries, e.g.:

- `TargetLowering` methods:
  - `getValueType(DL, Ty)` → `EVT`
  - `getRegisterType(...)` / `getNumRegisters(...)`
- Target hooks for:
  - Promoting/demoting types (`getTypeToPromoteTo`, etc.).
  - Deciding legal vector widths/types.

