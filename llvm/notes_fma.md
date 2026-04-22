# FMA 
**Fused Multiply-Add (FMA)** is a floating-point operation that computes `a × b + c` with improved accuracy and performance compared to separate multiply and add operations. The key benefit is **single rounding**: the intermediate product `a × b` is computed with full precision (infinite mantissa bits conceptually), then rounded only once when adding `c`.

**Benefits:**
- **Accuracy**: Single rounding vs. double rounding (multiply then add separately)
- **Performance**: Single instruction latency (typically 4-5 cycles) vs. two instructions (5-7+ cycles total)
- **Throughput**: Better instruction-level parallelism on modern CPUs

**LLVM Support:**
- IR intrinsics: `llvm.fma.*`, `llvm.fmuladd.*`
- DAG opcodes: `ISD::FMA`, `ISD::FMAD`, `ISD::FMULADD`
- Target instructions: FMA3 (Intel), FMA4 (AMD), AVX512-FMA (Intel/AMD)

---
## Hardware Support
### x86/x86-64 Architecture
#### FMA3 (Intel AVX2, Haswell 2013+)
**Subtarget Feature:** `FeatureFMA` in [`llvm/lib/Target/X86/X86.td:198`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86.td#L198)

**Supported Processors:**
- Intel: Haswell, Broadwell, Skylake, Cascade Lake, Ice Lake, Tiger Lake, Alder Lake, etc.
- AMD: Piledriver, Zen, Zen 2, Zen 3, Zen 4, etc.

**Instruction Forms:**
- **Scalar:** `VFMADD132SS/SD`, `VFMADD213SS/SD`, `VFMADD231SS/SD`
- **Packed 128-bit:** `VFMADD132PS/PD`, `VFMADD213PS/PD`, `VFMADD231PS/PD`
- **Packed 256-bit:** `VFMADD132PS/PDY`, `VFMADD213PS/PDY`, `VFMADD231PS/PDY`
- **Negative forms:** `VFNMADD*`, `VFNMSUB*`, `VFMSUB*`
- **Alternating:** `VFMADDSUB*`, `VFMSUBADD*`

**Operand Encoding (3-operand destructive form):**
```
Form 132: dst = src1 * src3 + src2   (dst = src1 before execution)
Form 213: dst = src1 * src2 + src3   (dst = src1 before execution)
Form 231: dst = src2 * src3 + src1   (dst = src1 before execution)
```

The form numbers indicate operand order for the computation: `1 * 3 + 2` → 132.

**Source:** [`llvm/lib/Target/X86/X86InstrFMA.td:18-34`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td#L18-L34)

#### FMA4 (AMD Bulldozer/Piledriver 2011-2012, Legacy)
**Subtarget Feature:** `FeatureFMA4` in [`llvm/lib/Target/X86/X86.td:201`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86.td#L201)

**Supported Processors:**
- AMD: Bulldozer, Piledriver

**Instruction Forms:**
- **Scalar:** `VFMADDSS4`, `VFMSUBSS4`, `VFMADDSD4`, `VFMSUBSD4`
- **Packed:** `VFMADDPS4`, `VFMSUBPS4`, `VFMADDPD4`, `VFMSUBPD4`

**Operand Encoding (4-operand non-destructive form):**
```
dst = src1 * src2 + src3  (dst can be any register)
```

**Source:** [`llvm/lib/Target/X86/X86InstrFMA.td:386-595`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td#L386-L595)

**Note:** FMA4 is deprecated; modern AMD processors use FMA3.

#### AVX-512 FMA Extensions
**Features:** Masking (k-registers), embedded broadcast, 512-bit vectors (ZMM registers)

**Masking Modes:**
- **K-merge-masked:** FMA result blended with original `src1` based on mask
- **K-zero-masked:** FMA result zeroed for masked-off elements

**Source:** [`llvm/lib/Target/X86/X86InstrAVX512.td:13782`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrAVX512.td#L13782)

---

## IR Level Representation

### Intrinsics

#### `llvm.fma.*` (Standard FMA)
```llvm
declare float @llvm.fma.f32(float %a, float %b, float %c)
declare double @llvm.fma.f64(double %a, double %b, double %c)
declare <4 x float> @llvm.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
declare <2 x double> @llvm.fma.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
```

**Semantics:** Computes `a × b + c` with **no intermediate rounding**. Single rounding at the end.

**Source:** [`llvm/include/llvm/IR/Intrinsics.td:1104-1106`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td#L1104-L1106)

#### `llvm.fmuladd.*` (Flexible Contraction)
```llvm
declare float @llvm.fmuladd.f32(float %a, float %b, float %c)
declare double @llvm.fmuladd.f64(double %a, double %b, double %c)
```

**Semantics:** Computes `a × b + c` with **implementation-defined rounding**. May use FMA (single rounding) or separate multiply-add (double rounding) depending on target capabilities and optimization settings.

**Source:** [`llvm/include/llvm/IR/Intrinsics.td:1107-1109`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td#L1107-L1109)

#### `llvm.experimental.constrained.fma.*` (Constrained FMA)
```llvm
declare float @llvm.experimental.constrained.fma.f32(
    float %a, float %b, float %c,
    metadata %rounding, metadata %except)
```

**Semantics:** Constrained FMA with explicit rounding mode and exception behavior control for strict IEEE-754 compliance.

**Source:** [`llvm/include/llvm/IR/Intrinsics.td:1289-1291`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td#L1289-L1291)

#### `llvm.vp.fma.*` (Vector Predicated FMA)
```llvm
declare <vscale x 4 x float> @llvm.vp.fma.nxv4f32(
    <vscale x 4 x float> %a,
    <vscale x 4 x float> %b,
    <vscale x 4 x float> %c,
    <vscale x 4 x i1> %mask,
    i32 %evl)
```

**Semantics:** Vector-predicated FMA with explicit vector length (EVL) and mask for scalable vectors.

**Source:** [`llvm/include/llvm/IR/Intrinsics.td:2248-2251`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td#L2248-L2251)

---

## Compiler Flags and Control

### Fast-Math Flags (FMF)

Fast-math flags control floating-point optimization aggressiveness. Defined in [`llvm/include/llvm/IR/FMF.h:34-42`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/FMF.h#L34-L42).

| Flag | Bit | Meaning | Impact on FMA |
|------|-----|---------|---------------|
| `nnan` | 1 | No NaNs assumed | Enables more aggressive FMA transformations (e.g., `fma(0, x, y) → y`) |
| `ninf` | 2 | No infinities assumed | Combines with `nnan` for zero-operand elimination |
| `nsz` | 3 | No signed zeros | Allows sign-insensitive optimizations |
| `arcp` | 4 | Allow reciprocal | Not directly FMA-related |
| `contract` | 5 | **Allow contraction** | **CRITICAL**: Enables `fadd(fmul(a,b),c) → fma(a,b,c)` |
| `afn` | 6 | Approximate functions | Not directly FMA-related |
| `reassoc` | 0 | Allow reassociation | Enables nested FMA combining: `fadd(fma(...), fmul(...)) → fma(..., fma(...))` |
| `fast` | - | All above flags | Maximum FMA generation |

**Example Usage in IR:**
```llvm
%mul = fmul fast float %a, %b          ; All fast-math flags
%add = fadd contract float %mul, %c    ; Only contract flag
%result = fadd nnan ninf contract float %x, %y
```

**Source:** [`llvm/include/llvm/IR/FMF.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/FMF.h)

### Floating-Point Operation Fusion Mode

Global compilation-wide setting controlling FP contraction. Defined in [`llvm/include/llvm/Target/TargetOptions.h:30-35`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetOptions.h#L30-L35).

```cpp
namespace llvm {
  enum class FPOpFusion {
    Fast,      // Aggressively fuse FP ops (e.g., fadd+fmul → fma)
    Standard,  // Only fuse blessed operations (fmuladd intrinsic)
    Strict     // Never fuse FP operations
  };
}
```

**Clang Frontend Mapping:**
- `-ffp-contract=off` → `FPOpFusion::Strict`
- `-ffp-contract=on` → `FPOpFusion::Standard` (default)
- `-ffp-contract=fast` → `FPOpFusion::Fast`

**Source:** [`llvm/include/llvm/Target/TargetOptions.h:30-35`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetOptions.h#L30-L35)

### Contraction Decision Logic

FMA contraction occurs when **any** of these conditions hold:

```cpp
// Source: llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17689-17693
// https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17689-L17693
bool AllowFusion = 
    (TLI.AllowFPOpFusion == FPOpFusion::Fast) ||    // Global flag
    HasFMAD ||                                       // Target has FMAD support
    N->getFlags().hasAllowContract();                // Node has 'contract' flag
```

**Additional Requirements:**
1. Target implements `isFMAFasterThanFMulAndFAdd(VT)` returning true
2. No conflicting constraints (e.g., strict FP mode)
3. Operands have compatible types

**Source:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17689-17707`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17689-L17707)

---

## DAG Opcodes and Semantics

### ISD::FMA
**Definition:** [`llvm/include/llvm/CodeGen/ISDOpcodes.h:517-518`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/ISDOpcodes.h#L517-L518)

```cpp
/// FMA - Perform a * b + c with no intermediate rounding step.
FMA,
```

**Semantics:**
- Computes `a * b + c` with **single rounding** at the end
- Infinite intermediate precision (conceptually)
- Result identical to IEEE-754 FMA operation
- **Rounding mode:** Respects current rounding mode (round-to-nearest-even by default)

**Properties:**
- **Commutative** in first two operands: `fma(a, b, c) == fma(b, a, c)`
- **Not associative:** `fma(fma(a,b,c),d,e) ≠ fma(a,b,fma(c,d,e))` in general
- **Not distributive:** `fma(a, b+c, d) ≠ fma(a,b,d) + fma(a,c,0)` in general

### ISD::FMAD
**Definition:** [`llvm/include/llvm/CodeGen/ISDOpcodes.h:520-521`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/ISDOpcodes.h#L520-L521)

```cpp
/// FMAD - Perform a * b + c, while getting the same result as the separately
/// rounded operations.
FMAD,
```

**Semantics:**
- Computes `a * b + c` with **intermediate rounding**
- Result must be **bit-identical** to `(a * b) + c` computed separately
- Less optimization potential than ISD::FMA
- Primarily for targets where hardware FMA respects intermediate rounding

**Note:** Most modern FMA hardware does **not** implement FMAD semantics; software emulation required.

### ISD::FMULADD
**Definition:** [`llvm/include/llvm/CodeGen/ISDOpcodes.h:523-527`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/ISDOpcodes.h#L523-L527)

```cpp
/// FMULADD - Perform a * b + c, while getting the same result as the
/// separately rounded operations. This is used for WebAssembly, which
/// doesn't have FMA but does have FMULADD.
FMULADD,
```

**Semantics:**
- Flexible semantics: may use FMA or separate operations
- Used for `llvm.fmuladd.*` intrinsic lowering
- Target decides optimal implementation

**Source:** [`llvm/include/llvm/CodeGen/ISDOpcodes.h:517-528`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/ISDOpcodes.h#L517-L528)

---

## FMA Generation and Contraction

### Automatic Contraction Patterns

FMA generation happens during **DAGCombine** phase. Key patterns:

#### Pattern 1: FADD + FMUL → FMA
```llvm
; IR
%mul = fmul fast float %a, %b
%add = fadd fast float %mul, %c

; DAG
(fadd (fmul a, b), c)

; Transformed to
(fma a, b, c)
```

**Implementation:** `visitFADDForFMACombine()` in [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17663-17896`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17663-L17896)

**Requirements:**
- `AllowContract` flag or `FPOpFusion::Fast`
- `FMUL` has exactly one use (avoid duplication)
- `isFMAFasterThanFMulAndFAdd()` returns true

#### Pattern 2: FSUB + FMUL → FMA (with negation)
```llvm
; IR
%mul = fmul fast float %a, %b
%sub = fsub fast float %mul, %c

; DAG
(fsub (fmul a, b), c)

; Transformed to
(fma a, b, (fneg c))
```

**Implementation:** `visitFSUBForFMACombine()` in [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17901-18120`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17901-L18120)

#### Pattern 3: Nested FMA Combining (with reassoc flag)
```llvm
; IR
%mul1 = fmul fast float %a, %b
%fma1 = call fast float @llvm.fma.f32(float %a, float %b, float %mul1)
%add = fadd reassoc fast float %fma1, %e

; DAG
(fadd (fma a, b, (fmul c, d)), e)

; Transformed to (with AllowReassociation)
(fma a, b, (fma c, d, e))
```

**Implementation:** `visitFADDForFMACombine()` inner loop at [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17748-17839`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17748-L17839)

**Requires:** `AllowReassociation` fast-math flag

#### Pattern 4: FP Extension Folding
```llvm
; IR
%ext_a = fpext half %a to float
%ext_b = fpext half %b to float
%mul = fmul fast float %ext_a, %ext_b
%add = fadd fast float %mul, %c

; DAG
(fadd (fmul (fpext a), (fpext b)), c)

; Transformed to
(fma (fpext a), (fpext b), c)
```

**Implementation:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17800-17839`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17800-L17839)

#### Pattern 5: FMA Distribution into FMUL
```llvm
; IR
%add1 = fadd fast float %x, 1.0
%mul = fmul fast float %add1, %y

; DAG
(fmul (fadd x, 1.0), y)

; Transformed to
(fma x, y, y)
```

**Implementation:** `visitFMULForFMADistributiveCombine()` in [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:18225-18304`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L18225-L18304)

**Note:** Enabled with `AllowReassociation` flag

---

## DAG Combining Transformations

### FADD Combining for FMA
**Function:** `visitFADDForFMACombine(SDNode *N)`  
**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17663-17896`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17663-L17896)

**Patterns Handled:**
1. `(fadd (fmul x, y), z)` → `(fma x, y, z)`
2. `(fadd z, (fmul x, y))` → `(fma x, y, z)` (commuted)
3. `(fadd (fma x, y, (fmul u, v)), z)` → `(fma x, y, (fma u, v, z))` (nested, requires reassoc)
4. `(fadd (fpext (fmul x, y)), z)` → `(fma (fpext x), (fpext y), z)` (extension folding)

**Key Checks:**
- Line 17689-17693: Contraction legality check
- Line 17694: `isFMAFasterThanFMulAndFAdd(VT)`
- Line 17699: FMUL one-use check (avoid duplication)
- Line 17705-17707: Prefer FMAD over FMA for precision when available
- Line 17748: `CanReassociate` check for nested combining

### FSUB Combining for FMA
**Function:** `visitFSUBForFMACombine(SDNode *N)`  
**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17901-18120`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17901-L18120)

**Patterns Handled:**
1. `(fsub (fmul x, y), z)` → `(fma x, y, (fneg z))`
2. `(fsub z, (fmul x, y))` → `(fma (fneg x), y, z)`
3. `(fsub (fneg (fmul x, y)), z)` → `(fma (fneg x), y, (fneg z))`
4. Double negation normalization

**Key Logic:**
- Line 17911-17919: Comments explain rounding semantics
- Line 17957-17973: Check for FNEG nodes to optimize negation patterns
- Line 17997-18036: Handle nested FMA with negation

### FMUL Distribution into FMA
**Function:** `visitFMULForFMADistributiveCombine(SDNode *N)`  
**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:18225-18304`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L18225-L18304)

**Patterns Handled:**
1. `(fmul (fadd x0, 1.0), y)` → `(fma x0, y, y)`
2. `(fmul (fadd x0, -1.0), y)` → `(fma x0, y, (fneg y))`
3. `(fmul (fsub 1.0, x1), y)` → `(fma (fneg x1), y, y)`
4. `(fmul (fsub -1.0, x1), y)` → `(fma (fneg x1), y, (fneg y))`

**Requirements:**
- AllowReassociation flag (changes operation order)
- Constant operand must be ±1.0
- Result: saves one ADD operation, uses FMA instead

**Note:** Line 18247-18248 warns: "Less precise result due to changed rounding order."

### FMA Node Optimization
**Function:** `visitFMA(SDNode *N)`  
**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:18877-19000`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L18877-L19000)

**Optimizations:**
1. **Constant folding** (Line 18883-18885):
   ```cpp
   if (SDValue Folded = foldConstantFPMath(...)) return Folded;
   ```

2. **Negation normalization** (Line 18895-18909):
   ```cpp
   // (-N0 * -N1) + N2 --> (N0 * N1) + N2
   // (-N0 * N1) + -N2 --> -(N0 * N1 + N2)
   ```

3. **Zero operand elimination** (Line 18911-18918):
   ```cpp
   // fma(0, x, y) -> y (with NoNaNs + NoInfs)
   // fma(x, 0, y) -> y (with NoNaNs + NoInfs)
   ```

4. **Constant coefficient combining** (Line 18925-18959):
   ```cpp
   // fma(x, c1, (fmul x, c2)) -> fmul(x, c1+c2)
   // fma(x, c1, (fadd (fmul x, c2), y)) -> fadd(fmul(x, c1+c2), y)
   ```

5. **Folding into memory operands** (Line 18976-18996):
   ```cpp
   // fma(a, b, load(addr)) -> fma_mem(a, b, addr)
   ```

### FMAD Node Optimization
**Function:** `visitFMAD(SDNode *N)`  
**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:19002-19014`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L19002-L19014)

**Optimizations:**
- **Only constant folding** — preserves intermediate rounding
- No aggressive transformations due to strict semantics

---

## X86 Target-Specific Implementation

### Target Hooks and Lowering

#### Legalization
**File:** [`llvm/lib/Target/X86/X86ISelLowering.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp)

**Lines 1611-1615:** FMA legalization setup
```cpp
if (Subtarget.hasAnyFMA()) {
  for (auto VT : {MVT::f32, MVT::f64, MVT::v4f32, MVT::v2f64, MVT::v8f32, MVT::v4f64}) {
    setOperationAction(ISD::FMA, VT, Legal);
    setOperationAction(ISD::STRICT_FMA, VT, Legal);
  }
}
```

**Lines 826-828:** Expansion when FMA unavailable
```cpp
setOperationAction(ISD::FMA, VT, Expand);  // Expands to fmul + fadd
```

**Lines 910-911:** Soft-float handling
```cpp
setOperationAction(ISD::FMA, MVT::f128, LibCall);
setLibcallName(RTLIB::FMA_F128, "fmal");
```

#### isFMAFasterThanFMulAndFAdd
**Function:** `X86TargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF, EVT VT)`  
**Location:** [`llvm/lib/Target/X86/X86ISelLowering.cpp:36156-36185`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp#L36156-L36185)

**Logic:**
```cpp
bool X86TargetLowering::isFMAFasterThanFMulAndFAdd(...) const {
  if (!Subtarget.hasAnyFMA())
    return false;
  
  VT = VT.getScalarType();
  
  if (!VT.isSimple())
    return false;
  
  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f16:
    return Subtarget.hasFP16();
  case MVT::bf16:
    return Subtarget.hasBF16() || Subtarget.hasAVXNECONVERT();
  case MVT::f32:
  case MVT::f64:
    return true;
  default:
    break;
  }
  
  return false;
}
```

**Returns true for:**
- `f32` (float) — always when FMA available
- `f64` (double) — always when FMA available  
- `f16` (half) — only with FP16 support
- `bf16` (bfloat16) — only with BF16 or AVXNECONVERT support

#### FMADDSUB/FMSUBADD Detection
**Function:** `isFMAddSubOrFMSubAdd(const X86Subtarget &Subtarget, SDValue &Opnd0, SDValue &Opnd1)`  
**Location:** [`llvm/lib/Target/X86/X86ISelLowering.cpp:8863-8886`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp#L8863-L8886)

**Purpose:** Detect patterns for `VFMADDSUB` / `VFMSUBADD` instructions

**Pattern:**
```llvm
; IR
%mul = fmul fast <4 x float> %a, %b
%sub = fsub fast <4 x float> %mul, %c    ; lanes 0, 2
%add = fadd fast <4 x float> %mul, %c    ; lanes 1, 3
%result = shufflevector %sub, %add, <i32 0, i32 5, i32 2, i32 7>

; DAG detects alternating add/sub pattern and generates:
(X86ISD::FMADDSUB a, b, c)

; Instruction
vfmaddsubps %xmm2, %xmm1, %xmm0, %xmm0
```

**Requirements:**
- FMUL has specific use count
- AllowContract flag present
- Alternating lanes follow ADDSUB or SUBADD pattern

**Source:** [`llvm/lib/Target/X86/X86ISelLowering.cpp:8839-8931`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp#L8839-L8931)

### Instruction Definitions

#### FMA3 TableGen Definitions
**File:** [`llvm/lib/Target/X86/X86InstrFMA.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td)

**Multiclass Structure:**

```tablegen
// Lines 36-54: fma3p_rm_213 (213 form packed)
multiclass fma3p_rm_213<bits<8> opc, string OpcodeStr, RegisterClass RC,
                        ValueType VT, X86MemOperand x86memop, PatFrag MemFrag,
                        SDPatternOperator Op, X86FoldableSchedWrite sched> {
  def r : FMA3<opc, MRMSrcReg, (outs RC:$dst),
               (ins RC:$src1, RC:$src2, RC:$src3),
               !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
               [(set RC:$dst, (VT (Op RC:$src2, RC:$src1, RC:$src3)))]>,
               Sched<[sched]>;
  
  let mayLoad = 1 in
  def m : FMA3<opc, MRMSrcMem, (outs RC:$dst),
               (ins RC:$src1, RC:$src2, x86memop:$src3),
               !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
               [(set RC:$dst, (VT (Op RC:$src2, RC:$src1, (MemFrag addr:$src3))))]>,
               Sched<[sched.Folded, sched.ReadAfterFold, sched.ReadAfterFold]>;
}
```

**Key Attributes (Line 98):**
```tablegen
let Constraints = "$src1 = $dst",  // Destructive encoding
    hasSideEffects = 0,
    isCommutable = 1,              // Operands commutable (with opcode change)
    Uses = [MXCSR],                // Uses FP control/status register
    mayRaiseFPException = 1        // May raise IEEE exceptions
```

**Instruction Instances (Lines 124-166):**

| Instruction Family | Opcode 132 | Opcode 213 | Opcode 231 | Data Type | Operation |
|--------------------|------------|------------|------------|-----------|-----------|
| `VFMADD*PS` | 0x98 | 0xA8 | 0xB8 | Packed Single | `a*b+c` |
| `VFMADD*PD` | 0x98 | 0xA8 | 0xB8 | Packed Double | `a*b+c` |
| `VFMSUB*PS` | 0x9A | 0xAA | 0xBA | Packed Single | `a*b-c` |
| `VFMSUB*PD` | 0x9A | 0xAA | 0xBA | Packed Double | `a*b-c` |
| `VFNMADD*PS` | 0x9C | 0xAC | 0xBC | Packed Single | `-(a*b)+c` |
| `VFNMADD*PD` | 0x9C | 0xAC | 0xBC | Packed Double | `-(a*b)+c` |
| `VFNMSUB*PS` | 0x9E | 0xAE | 0xBE | Packed Single | `-(a*b)-c` |
| `VFNMSUB*PD` | 0x9E | 0xAE | 0xBE | Packed Double | `-(a*b)-c` |
| `VFMADDSUB*PS` | 0x96 | 0xA6 | 0xB6 | Packed Single | Alternating |
| `VFMADDSUB*PD` | 0x96 | 0xA6 | 0xB6 | Packed Double | Alternating |
| `VFMSUBADD*PS` | 0x97 | 0xA7 | 0xB7 | Packed Single | Alternating |
| `VFMSUBADD*PD` | 0x97 | 0xA7 | 0xB7 | Packed Double | Alternating |

**Scalar Instructions ([Lines 307-330](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td#L307-L330)):**
- `VFMADD*SS` / `VFMADD*SD` (scalar single/double)
- `VFMSUB*SS` / `VFMSUB*SD`
- `VFNMADD*SS` / `VFNMADD*SD`
- `VFNMSUB*SS` / `VFNMSUB*SD`

#### FMA4 TableGen Definitions
**File:** [`llvm/lib/Target/X86/X86InstrFMA.td:386-633`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td#L386-L633)

**Key Differences from FMA3:**
- 4-operand non-destructive form
- Separate register classes for source operands
- Older AMD-specific encoding

**Example (Lines 540-552):**
```tablegen
defm VFMADDSS4  : fma4s<0x6A, "vfmaddss", FR32, f32mem, f32, any_fma, loadf32,
                        SchedWriteFMA.Scl>,
                  fma4s_int<0x6A, "vfmaddss", ssmem, SchedWriteFMA.Scl>;
```

### FMA3 Opcode Information
**File:** [`llvm/lib/Target/X86/X86InstrFMA3Info.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA3Info.h)

**Structure ([Lines 21-52](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA3Info.h#L21-L52)):**
```cpp
struct X86InstrFMA3Group {
  // 132 form: result = src1 * src3 + src2
  uint16_t Form132;
  // 213 form: result = src1 * src2 + src3
  uint16_t Form213;
  // 231 form: result = src2 * src3 + src1
  uint16_t Form231;
  
  // Attributes
  bool IsIntrinsic;
  bool HasKMergeMasked;   // AVX-512 masking support
  bool HasKZeroMasked;    // AVX-512 zero-masking support
};
```

**Opcode Groups Table ([Lines 66-97](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA3Info.h#L66-L97)):**
Arrays `X86InstrFMA3Groups[]` contain all FMA3 opcode combinations for efficient commutation and form conversion.

---

## Instruction Selection

### Pattern Matching

#### DAG Patterns (SelectionDAG)
**File:** [`llvm/lib/Target/X86/X86InstrFMA.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td)

**Example Pattern (Lines 43-44):**
```tablegen
def r : FMA3<opc, MRMSrcReg, (outs RC:$dst),
             (ins RC:$src1, RC:$src2, RC:$src3),
             "vfmadd213ps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
             [(set RC:$dst, (VT (Op RC:$src2, RC:$src1, RC:$src3)))]>,
             Sched<[sched]>;
```

**Pattern Components:**
- `Op` → `any_fma` (defined in [`llvm/include/llvm/Target/TargetSelectionDAG.td:1804`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetSelectionDAG.td#L1804))
- `any_fma` → `PatFrags<(ops node:$src1, node:$src2, node:$src3), [(fma node:$src1, node:$src2, node:$src3), (strict_fma node:$src1, node:$src2, node:$src3)]>`
- Matches both `ISD::FMA` and `ISD::STRICT_FMA`

**Memory Folding Pattern (Lines 47-53):**
```tablegen
let mayLoad = 1 in
def m : FMA3<opc, MRMSrcMem, (outs RC:$dst),
             (ins RC:$src1, RC:$src2, x86memop:$src3),
             "vfmadd213ps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
             [(set RC:$dst, (VT (Op RC:$src2, RC:$src1, (MemFrag addr:$src3))))]>,
             Sched<[sched.Folded, sched.ReadAfterFold, sched.ReadAfterFold]>;
```

**Third operand loaded from memory** — reduces register pressure.

### Commutation and Form Selection

#### Commutation Rules
**Source:** [`llvm/lib/Target/X86/X86InstrFMA.td:18-34`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td#L18-L34)

FMA3 instructions are **isCommutable = 1**, meaning TableGen/CodeGen can swap operands with appropriate opcode changes.

**213 Form Commutation:**
- Swap ops 1 ↔ 2: No opcode change (stays 213)
- Swap ops 1 ↔ 3: Change to 231 form
- Swap ops 2 ↔ 3: Change to 132 form

**132 Form Commutation:**
- Swap ops 1 ↔ 2: Change to 231 form
- Swap ops 1 ↔ 3: No opcode change (stays 132)
- Swap ops 2 ↔ 3: Change to 213 form

**231 Form Commutation:**
- Swap ops 1 ↔ 2: Change to 132 form
- Swap ops 1 ↔ 3: Change to 213 form
- Swap ops 2 ↔ 3: No opcode change (stays 231)

#### Form Selection Heuristics

Register allocator and instruction scheduler prefer specific forms based on:

1. **Register lifetime:** Minimize live ranges by destructively updating earlier-dying operand
2. **Memory operand availability:** Choose form that allows memory folding
3. **Dependency chains:** Select form reducing critical path latency

**Example:**
```assembly
; If %xmm0 dies here, prefer 213 form:
vfmadd213ps  %xmm2, %xmm1, %xmm0   ; %xmm0 = %xmm1 * %xmm0 + %xmm2

; If %xmm2 can be folded from memory:
vfmadd213ps  (%rdi), %xmm1, %xmm0  ; %xmm0 = %xmm1 * %xmm0 + [%rdi]
```

### AVX-512 Masking Support

AVX-512 FMA instructions support **predicate masking** via k-registers.

**Example:**
```assembly
; K-merge-masked: blend FMA result with original src1 based on k1 mask
vfmadd213ps %zmm2, %zmm1, %zmm0 {%k1}

; K-zero-masked: zero masked-off elements
vfmadd213ps %zmm2, %zmm1, %zmm0 {%k1}{z}
```

**TableGen Support:** [`llvm/lib/Target/X86/X86InstrAVX512.td:13782`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrAVX512.td#L13782)

---

## Optimization Properties

### Commutative Properties
FMA is **commutative in the first two operands** (multiply operands):

```
fma(a, b, c) == fma(b, a, c)
```

**Not commutative with third operand:**
```
fma(a, b, c) ≠ fma(a, c, b)  // Generally
```

**Source:** [`llvm/include/llvm/Target/TargetSelectionDAG.td:569`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetSelectionDAG.td#L569)
```tablegen
def fma : SDNode<"ISD::FMA", SDTFPTernaryOp, [SDNPCommutative]>;
```

The `SDNPCommutative` property applies to the entire node (considering operands 0 and 1).

### Associativity and Reassociation

FMA is **not associative**:
```
fma(fma(a, b, c), d, e) ≠ fma(a, b, fma(c, d, e))
```

However, with `AllowReassociation` flag, LLVM may perform associative transformations assuming acceptable precision loss.

**Example Transformation (Lines 17748-17839):**
```cpp
// (fadd (fma x, y, (fmul u, v)), z)
// With AllowReassociation:
// → (fma x, y, (fma u, v, z))
```

**Benefit:** Reduces dependency chain depth, improves instruction-level parallelism.

### Precision Considerations

#### Single vs. Double Rounding

**FMA (ISD::FMA):**
- **Single rounding:** `a × b` computed with infinite precision, then `+ c` rounded once
- **More accurate** than separate operations

**FMAD (ISD::FMAD):**
- **Double rounding:** `a × b` rounded, then `+ c` rounded again
- **Same precision** as separate `fmul` + `fadd`

**Example Where FMA Differs:**
```c
// a = 1.0 + 2^-53 (smallest float64 increment at 1.0)
// b = 1.0
// c = -1.0

// Separate operations (double rounding):
//   a * b = 1.0 + 2^-53 (exact)
//   (a * b) + c = (1.0 + 2^-53) + (-1.0) = 2^-53 (rounds to 2^-53)

// FMA (single rounding):
//   fma(a, b, c) = a*b + c = (1.0 + 2^-53)*1.0 + (-1.0)
//                = 1.0 + 2^-53 - 1.0 = 2^-53 (exact result, no rounding needed)
```

In most cases, FMA produces **more accurate** results due to avoiding intermediate rounding error accumulation.

#### Rounding Order Changes

**FMUL Distributive Transformation (Lines 18247-18248):**
```cpp
// (fmul (fadd x, 1.0), y) → (fma x, y, y)
// WARNING: Less precise result due to changed rounding order.
```

**Why less precise?**
- Original: `(x + 1.0)` rounds, then `* y` rounds → two roundings
- FMA: `x * y` computed with full precision, then `+ y` rounds → one rounding, but different order
- Different rounding order can produce different results

### Constant Folding

**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:18883-18885`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L18883-L18885)

```cpp
SDValue Folded = foldConstantFPMath(ISD::FMA, DL, VT, {N0, N1, N2});
if (Folded) return Folded;
```

**Example:**
```llvm
; Compile-time evaluation
%result = call float @llvm.fma.f32(float 2.0, float 3.0, float 5.0)
; Constant folded to:
%result = float 11.0
```

### Negation Normalization

**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:18895-18909`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L18895-L18909)

**Patterns:**
1. `fma(-a, -b, c)` → `fma(a, b, c)` (double negation cancels)
2. `fma(-a, b, -c)` → `-(fma(a, b, c))` (factor out negation)
3. `fma(a, -b, -c)` → `-(fma(a, b, c))`

**Benefit:** Reduces FNEG instruction count, enables better pattern matching.

### Zero and Identity Elimination

**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:18911-18918`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L18911-L18918)

**With NoNaNs + NoInfs flags:**
```cpp
// fma(0, x, y) → y
// fma(x, 0, y) → y
// fma(x, y, 0) → x * y
```

**Without NoNaNs + NoInfs:**
These transformations are **invalid** because:
- `fma(0, Inf, y)` → NaN (not y)
- `fma(NaN, x, y)` → NaN (not y)

### Coefficient Combining

**Location:** [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:18925-18959`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L18925-L18959)

**Pattern:**
```cpp
// fma(x, c1, (fmul x, c2)) → fmul(x, c1+c2)
```

**Example:**
```llvm
%mul1 = fmul fast float %x, 3.0
%fma = call fast float @llvm.fma.f32(float %x, float 5.0, float %mul1)
; Optimized to:
%result = fmul fast float %x, 8.0   ; 3.0 + 5.0 = 8.0
```

**Benefit:** Reduces instruction count from FMA + FMUL to single FMUL.

---

## Preconditions and Postconditions

### Preconditions for FMA Contraction

#### Global Preconditions
1. **Target Support:**
   - `Subtarget.hasAnyFMA()` returns true (X86: FMA3 or FMA4)
   - `isFMAFasterThanFMulAndFAdd(VT)` returns true
   - Legal action set for `ISD::FMA` on value type VT

2. **Compilation Mode:**
   - `FPOpFusion == FPOpFusion::Fast`, OR
   - `FPOpFusion == FPOpFusion::Standard` AND operation is `llvm.fmuladd.*`, OR
   - Node has `AllowContract` fast-math flag

#### Pattern-Specific Preconditions

**FADD + FMUL → FMA:**
```cpp
// Source: llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:17699-17701
// https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17699-L17701
1. FMUL has exactly one use (avoid instruction duplication)
2. AllowFusion == true (contraction permitted)
3. VT is legal FMA type
```

**Nested FMA Combining:**
```cpp
// Line 17748
// https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17748
1. AllowReassociation flag set (operation order may change)
2. Inner operation is FMUL with one use, OR
3. Inner operation is FMA
```

**FP Extension Folding:**
```cpp
// Lines 17800-17839
// https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17800-L17839
1. Both multiply operands have FPEXT nodes
2. Extensions from same source type
3. Extension legality check passes
```

### Postconditions After FMA Generation

#### Properties Preserved
1. **Fast-Math Flags:**
   - All input flags propagated to FMA node
   - `SelectionDAG::FlagInserter` ensures consistency (Lines 17887, 18925)

2. **Value Range:**
   - FMA result numerically close to original `(a*b)+c`
   - Precision typically better (single rounding)

3. **Exception Behavior:**
   - IEEE-754 exception semantics maintained
   - Invalid, overflow, underflow, inexact flags raised appropriately

#### Properties Changed
1. **Rounding:**
   - Single rounding instead of double rounding
   - May produce different bit-exact result

2. **Instruction Count:**
   - Two instructions (FMUL + FADD) → one instruction (FMA)

3. **Latency:**
   - Reduced dependency chain depth
   - Typically 4-5 cycles (FMA) vs. 4+3=7 cycles (FMUL + FADD)

4. **Register Pressure:**
   - Intermediate FMUL result eliminated
   - One fewer live value

---

## Code Examples

### Example 1: Basic FMA Contraction

**C Code:**
```c
float fma_example(float a, float b, float c) {
    return a * b + c;
}
```

**Clang Command:**
```bash
clang -O2 -march=haswell -ffp-contract=fast -S -emit-llvm fma.c -o -
```

**LLVM IR:**
```llvm
define float @fma_example(float %a, float %b, float %c) {
  %mul = fmul fast float %a, %b
  %add = fadd fast float %mul, %c
  ret float %add
}
```

**DAG (before combining):**
```
t2: f32 = fmul t0, t1
t3: f32 = fadd t2, t3
```

**DAG (after combining):**
```
t2: f32 = fma t0, t1, t3
```

**x86-64 Assembly:**
```assembly
fma_example:
    vfmadd213ss  %xmm2, %xmm1, %xmm0  ; %xmm0 = %xmm0 * %xmm1 + %xmm2
    ret
```

### Example 2: Nested FMA with Reassociation

**C Code:**
```c
float nested_fma(float a, float b, float c, float d, float e) {
    return a * b + (c * d + e);
}
```

**Clang Command:**
```bash
clang -O3 -march=haswell -ffp-contract=fast -ffast-math -S nested.c -o -
```

**LLVM IR:**
```llvm
define float @nested_fma(float %a, float %b, float %c, float %d, float %e) {
  %mul1 = fmul reassoc fast float %a, %b
  %mul2 = fmul reassoc fast float %c, %d
  %add1 = fadd reassoc fast float %mul2, %e
  %add2 = fadd reassoc fast float %mul1, %add1
  ret float %add2
}
```

**DAG (after combining):**
```
t6: f32 = fma t4, t5, t7        ; c * d + e
t8: f32 = fma t0, t1, t6        ; a * b + (fma result)
```

**x86-64 Assembly:**
```assembly
nested_fma:
    vfmadd213ss  %xmm4, %xmm3, %xmm2  ; c*d + e
    vfmadd213ss  %xmm2, %xmm1, %xmm0  ; a*b + prev
    ret
```

### Example 3: FMADDSUB Pattern

**C Code:**
```c
#include <immintrin.h>

__m128 fmaddsub_example(__m128 a, __m128 b, __m128 c) {
    __m128 mul = _mm_mul_ps(a, b);
    __m128 sub = _mm_sub_ps(mul, c);  // lanes 0, 2
    __m128 add = _mm_add_ps(mul, c);  // lanes 1, 3
    return _mm_shuffle_ps(sub, add, _MM_SHUFFLE(3, 1, 2, 0));
}
```

**LLVM IR:**
```llvm
define <4 x float> @fmaddsub_example(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
  %mul = fmul fast <4 x float> %a, %b
  %sub = fsub fast <4 x float> %mul, %c
  %add = fadd fast <4 x float> %mul, %c
  %shuffle = shufflevector <4 x float> %sub, <4 x float> %add, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %shuffle
}
```

**DAG (after X86 pattern detection):**
```
t5: v4f32 = X86ISD::FMADDSUB t0, t1, t2
```

**x86-64 Assembly:**
```assembly
fmaddsub_example:
    vfmaddsubps  %xmm2, %xmm1, %xmm0, %xmm0
    ret
```

**Result:** Single instruction instead of four (fmul, fsub, fadd, shuffle).

### Example 4: FMA with Memory Operand

**C Code:**
```c
float fma_mem(float a, float b, const float *c) {
    return a * b + *c;
}
```

**x86-64 Assembly (with memory folding):**
```assembly
fma_mem:
    vfmadd213ss  (%rdi), %xmm1, %xmm0  ; %xmm0 = %xmm0 * %xmm1 + [%rdi]
    ret
```

**Benefit:** No register allocated for `*c` — loaded directly as memory operand.

### Example 5: Avoiding FMA (Strict Mode)

**C Code:**
```c
// Compile with -ffp-contract=off
double no_fma(double a, double b, double c) {
    return a * b + c;  // Must use separate mul + add
}
```

**Clang Command:**
```bash
clang -O2 -march=haswell -ffp-contract=off -S no_fma.c -o -
```

**x86-64 Assembly:**
```assembly
no_fma:
    vmulsd   %xmm1, %xmm0, %xmm0   ; Separate multiply
    vaddsd   %xmm2, %xmm0, %xmm0   ; Separate add
    ret
```

**Use Case:** When bit-exact reproducibility across architectures required.

### Example 6: Constrained FMA (Strict FP)

**LLVM IR:**
```llvm
define float @strict_fma(float %a, float %b, float %c) #0 {
  %result = call float @llvm.experimental.constrained.fma.f32(
                float %a, float %b, float %c,
                metadata !"round.tonearest",
                metadata !"fpexcept.strict")
  ret float %result
}

attributes #0 = { strictfp }
```

**DAG:**
```
t2: f32 = strict_fma t0, t1, t3, TargetConstant:i32<12>
```

**x86-64 Assembly:**
```assembly
strict_fma:
    vfmadd213ss  %xmm2, %xmm1, %xmm0
    ret
```

**Difference:** Exception behavior strictly follows IEEE-754, no reordering allowed.

---


### LLVM Core

| File | Purpose | Key Lines |
|------|---------|-----------|
| [`llvm/include/llvm/CodeGen/ISDOpcodes.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/ISDOpcodes.h) | ISD opcode definitions | [517-528](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/ISDOpcodes.h#L517-L528) |
| [`llvm/include/llvm/IR/Intrinsics.td`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td) | FMA intrinsic declarations | [1104-1109](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td#L1104-L1109), [1289-1291](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td#L1289-L1291), [2248-2251](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td#L2248-L2251) |
| [`llvm/include/llvm/IR/FMF.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/FMF.h) | Fast-math flag definitions | [34-42](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/FMF.h#L34-L42), [72](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/FMF.h#L72), [93](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/FMF.h#L93) |
| [`llvm/include/llvm/Target/TargetOptions.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetOptions.h) | FPOpFusion enum | [30-35](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetOptions.h#L30-L35) |
| [`llvm/include/llvm/Target/TargetSelectionDAG.td`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetSelectionDAG.td) | DAG node definitions | [145](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetSelectionDAG.td#L145), [569-570](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetSelectionDAG.td#L569-L570) |
| [`llvm/include/llvm/CodeGen/TargetLowering.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/TargetLowering.h) | Target lowering hooks | [980-1025](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/TargetLowering.h#L980-L1025), [3439](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/CodeGen/TargetLowering.h#L3439) |
| [`llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp) | FMA combining patterns | [17663-19025](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp#L17663-L19025) |
| [`llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp) | IR → DAG lowering | [7131-7157](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp#L7131-L7157), [8587-8591](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp#L8587-L8591) |

### X86 Target

| File | Purpose | Key Lines |
|------|---------|-----------|
| [`llvm/lib/Target/X86/X86ISelLowering.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp) | X86 FMA lowering logic | [609-2576](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp#L609-L2576), [8839-8931](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp#L8839-L8931), [36156-36185](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp#L36156-L36185) |
| [`llvm/lib/Target/X86/X86ISelLowering.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.h) | X86 lowering declarations | [48-1850](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.h#L48-L1850) |
| [`llvm/lib/Target/X86/X86InstrFMA.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td) | FMA instruction definitions | [1-633](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td#L1-L633) |
| [`llvm/lib/Target/X86/X86InstrFMA3Info.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA3Info.h) | FMA3 opcode group info | [21-97](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA3Info.h#L21-L97) |
| [`llvm/lib/Target/X86/X86InstrUtils.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrUtils.td) | FMA instruction classes | [832-886](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrUtils.td#L832-L886) |
| [`llvm/lib/Target/X86/X86.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86.td) | FMA feature flags | [198-206](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86.td#L198-L206) |
| [`llvm/lib/Target/X86/X86Subtarget.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86Subtarget.h) | Subtarget FMA queries | [210](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86Subtarget.h#L210) |
| [`llvm/lib/Target/X86/X86TargetTransformInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86TargetTransformInfo.cpp) | FMA cost modeling | Various |

---

## Summary

### Quick Reference

**FMA Contraction Decision Tree:**
```
Can FADD + FMUL be contracted to FMA?
├─ Does target support FMA? (hasAnyFMA())
│  ├─ NO → Expand to separate FMUL + FADD
│  └─ YES
│     └─ Is FMA faster? (isFMAFasterThanFMulAndFAdd())
│        ├─ NO → Keep separate FMUL + FADD
│        └─ YES
│           └─ Is contraction allowed?
│              ├─ FPOpFusion == Fast → YES
│              ├─ Node has AllowContract flag → YES
│              ├─ HasFMAD and operation is blessed → YES
│              └─ Otherwise → NO
```

**Key Flags:**
- **IR Level:** `fast`, `contract`, `reassoc` fast-math flags
- **Global:** `-ffp-contract=off/on/fast` (Clang), `FPOpFusion` enum
- **Target:** `FeatureFMA` (FMA3), `FeatureFMA4` (FMA4)

**Performance Impact:**
- **Latency:** ~4-5 cycles (FMA) vs. ~7 cycles (FMUL + FADD)
- **Throughput:** 2x FMAs/cycle (modern Intel/AMD) vs. 2x MUL + 2x ADD/cycle
- **Accuracy:** Single rounding (FMA) vs. double rounding (separate ops)

**Common Pitfalls:**
1. **Forgetting `-ffp-contract=fast`:** FMA won't generate without contraction flag
2. **Using `-ffp-contract=off` unintentionally:** Disables all FMA contraction
3. **Assuming associativity:** FMA is NOT associative without `reassoc` flag
4. **Ignoring precision changes:** FMA produces different bit-patterns than FMUL+FADD

**Best Practices:**
1. Use `-march=native` or explicit `-march=haswell` etc. for FMA support
2. Use `-ffp-contract=fast` for maximum FMA generation
3. Add `#pragma clang fp contract(fast)` for specific regions
4. Use `llvm.fma.*` intrinsic for guaranteed FMA (if supported)
5. Verify generated assembly (`-S -masm=intel`) to confirm FMA usage

---

## Glossary


#### FMA (Fused Multiply-Add)
**Definition:** A single floating-point operation that computes `a × b + c` with **single rounding** at the end.

**Key Properties:**
- Intermediate product `a × b` computed with infinite precision (conceptually)
- Only one rounding occurs when adding `c`
- More accurate than separate multiply and add
- Hardware implements this as a single instruction

**Mathematical Example:**
```
a = 1.0 + 2^-53 (smallest increment for double precision at 1.0)
b = 1.0
c = -1.0

Separate operations (double rounding):
  Step 1: a × b = 1.0000000000000002 (exact, no rounding needed)
  Step 2: result + c = 1.0000000000000002 + (-1.0) 
        = 0.0000000000000002 (rounds to 2^-53)

FMA (single rounding):
  Compute: (1.0 + 2^-53) × 1.0 + (-1.0)
         = 1.0 + 2^-53 - 1.0
         = 2^-53 (exact, no rounding needed)

Result is the same in this case, but FMA is guaranteed to never be less accurate.
```

**LLVM IR Example:**
```llvm
; Using FMA intrinsic
%result = call float @llvm.fma.f32(float %a, float %b, float %c)

; Without FMA (separate operations)
%mul = fmul float %a, %b
%result = fadd float %mul, %c
```

**C/C++ Example:**
```c
#include <math.h>

float with_fma(float a, float b, float c) {
    return fmaf(a, b, c);  // Guaranteed FMA if hardware supports it
}

float without_fma(float a, float b, float c) {
    return a * b + c;  // May or may not use FMA depending on flags
}
```

---

#### FMAD (Fused Multiply-Add with intermediate rounding)
**Definition:** A multiply-add operation that produces the **same result** as separate multiply and add operations (double rounding).

**Key Properties:**
- Must be bit-exact with `(a * b) + c`
- Respects intermediate rounding
- Rarely implemented in hardware
- Used when reproducibility with separate operations is required

**Comparison Example:**
```
Value: a = 1.5, b = 1.5, c = 0.1 (approximations in binary)

ISD::FMA (single rounding):
  - Compute a*b with infinite precision: 2.25 (exact)
  - Add c: 2.25 + 0.1 → round once → result
  
ISD::FMAD (double rounding):
  - Compute a*b: 1.5 * 1.5 → round → 2.25
  - Add c: 2.25 + 0.1 → round → result
  - Must match separate operations exactly

For most values FMA and FMAD differ!
```

**When to Use:**
- Cross-platform reproducibility required
- Testing against reference implementations using separate operations
- Verifying compiler correctness

---

#### FMULADD (Flexible FMA)
**Definition:** A multiply-add operation with **implementation-defined** rounding behavior. Compiler may choose FMA or separate operations.

**Key Properties:**
- Semantics: "compiler's choice"
- Used for `llvm.fmuladd.*` intrinsic
- Target decides based on performance/availability

**LLVM IR Example:**
```llvm
; Flexible - compiler chooses implementation
%result = call float @llvm.fmuladd.f32(float %a, float %b, float %c)

; On x86 with FMA3: generates vfmadd213ss (single rounding)
; On x86 without FMA: generates vmulss + vaddss (double rounding)
; On ARM with VFP: generates vmla (single rounding on some cores)
```

**C Example (from `<math.h>`):**
```c
// Standard C FMA - guaranteed single rounding
float std_fma = fmaf(a, b, c);

// Compiler-specific "fmuladd" - rounding not specified
// In Clang/LLVM, generated from:
float auto_fma = a * b + c;  // With -ffp-contract=on (default)
```

---

### Fast-Math Flags

#### contract (AllowContract)
**Definition:** Fast-math flag that permits **contraction** of floating-point expressions.

**Bit Position:** 5 in FMF (Fast-Math Flags)

**What it enables:**
```
fadd(fmul(a, b), c) → fma(a, b, c)  // Contraction allowed
```

**LLVM IR Syntax:**
```llvm
; With contract flag
%mul = fmul contract float %a, %b
%add = fadd contract float %mul, %c
; DAGCombiner can transform to:
; %result = fma float %a, %b, %c

; Without contract flag
%mul = fmul float %a, %b
%add = fadd float %mul, %c
; Stays as separate fmul + fadd (unless -ffp-contract=fast globally)
```

**Clang C/C++ Usage:**
```c
// Enable contraction for entire function
#pragma clang fp contract(fast)
float func1(float a, float b, float c) {
    return a * b + c;  // Will generate FMA
}

// Enable contraction for specific expression
float func2(float a, float b, float c) {
    #pragma clang fp contract(fast)
    return a * b + c;  // FMA here
}

// Disable contraction
#pragma clang fp contract(off)
float func3(float a, float b, float c) {
    return a * b + c;  // Separate multiply and add
}
```

**Command-line flags:**
```bash
# Allow contraction for blessed operations only (default)
clang -ffp-contract=on input.c

# Allow aggressive contraction everywhere
clang -ffp-contract=fast input.c

# Disable all contraction
clang -ffp-contract=off input.c
```

**Example Transformation:**
```llvm
; Before (with contract flag):
define float @example(float %a, float %b, float %c) {
  %mul = fmul contract float %a, %b
  %add = fadd contract float %mul, %c
  ret float %add
}

; After DAGCombine:
define float @example(float %a, float %b, float %c) {
  %fma = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %fma
}

; Final x86 assembly (with -march=haswell):
vfmadd213ss %xmm2, %xmm1, %xmm0
ret
```

**When contraction is NOT applied (even with flag):**
1. Target doesn't support FMA
2. FMA is slower than separate operations
3. `FMUL` has multiple uses (would duplicate computation)
4. Value types don't match

---

#### reassoc (AllowReassociation)
**Definition:** Fast-math flag that permits **reordering** and **regrouping** of floating-point operations.

**Bit Position:** 0 in FMF

**What it enables:**
```
(a + b) + c → a + (b + c)           // Reassociation
a + b + c + d → (a + c) + (b + d)   // Regrouping
fadd(fma(x,y,z), w) → fma(x, y, fadd(z, w))  // Nested FMA
```

**LLVM IR Example:**
```llvm
; Example 1: Simple reassociation
%1 = fadd reassoc float %a, %b
%2 = fadd reassoc float %1, %c
; Can transform to:
%tmp = fadd reassoc float %b, %c
%result = fadd reassoc float %a, %tmp

; Example 2: FMA nesting
%mul1 = fmul reassoc contract float %a, %b
%mul2 = fmul reassoc contract float %c, %d
%fma1 = call reassoc contract float @llvm.fma.f32(float %a, float %b, float %mul2)
%add = fadd reassoc contract float %fma1, %e
; Can transform to:
%inner = call float @llvm.fma.f32(float %c, float %d, float %e)
%result = call float @llvm.fma.f32(float %a, float %b, float %inner)
```

**C Example:**
```c
// Without reassoc - operations execute left-to-right strictly
float no_reassoc(float a, float b, float c, float d) {
    float t1 = a * b;
    float t2 = c * d;
    return t1 + t2;  // Must compute in this order
}

// With reassoc - compiler can reorder
float with_reassoc(float a, float b, float c, float d) 
    __attribute__((optnone)) {
    #pragma clang fp reassociate(on) contract(fast)
    return a * b + c * d;
    // Compiler might transform to:
    // fma(a, b, fma(c, d, 0))  // Nested FMAs
    // Or parallelize: (a*b) and (c*d) computed simultaneously
}
```

**Real-World Impact:**
```c
// Example: Dot product
float dot4(float *a, float *b) {
    #pragma clang fp reassociate(on) contract(fast)
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

// Without reassoc:
//   t1 = a[0]*b[0]
//   t2 = a[1]*b[1]
//   t3 = t1 + t2           // Dependency on t1, t2
//   t4 = a[2]*b[2]
//   t5 = t3 + t4           // Dependency on t3, t4
//   t6 = a[3]*b[3]
//   result = t5 + t6       // Dependency chain is long

// With reassoc:
//   t1 = a[0]*b[0]
//   t2 = a[1]*b[1]
//   t3 = a[2]*b[2]
//   t4 = a[3]*b[3]
//   s1 = t1 + t2           // Independent of s2
//   s2 = t3 + t4           // Can execute in parallel with s1
//   result = s1 + s2       // Shorter critical path!

// Or with FMA nesting:
//   result = fma(a[0], b[0], fma(a[1], b[1], fma(a[2], b[2], a[3]*b[3])))
```

**Clang Usage:**
```bash
# Enable reassociation
clang -ffast-math input.c          # Enables all, including reassoc
clang -funsafe-math-optimizations input.c  # Includes reassoc

# Individual flag (requires -Xclang)
clang -Xclang -mreassociate input.c
```

---

#### nnan (NoNaNs)
**Definition:** Fast-math flag asserting that **no NaN values** will occur in computation.

**What it enables:**
```c
fma(0.0, x, y) → y      // Safe only if x is never NaN or Inf
fma(x, 0.0, y) → y      // Safe only if x is never NaN or Inf
x + 0.0 → x             // Safe only if x is never NaN
```

**LLVM IR Example:**
```llvm
%result = fadd nnan float %x, 0.0
; Optimized to:
%result = %x  ; Addition eliminated

; But without nnan:
%result = fadd float %x, 0.0
; Cannot eliminate because if %x is NaN, result must be NaN
```

**Why it matters for FMA:**
```llvm
; With nnan + ninf:
%fma = call nnan ninf float @llvm.fma.f32(float 0.0, float %x, float %y)
; Optimized to:
%result = %y

; Without nnan:
; Cannot optimize because:
;   fma(0, Inf, y) = NaN (not y)
;   fma(0, NaN, y) = NaN (not y)
```

**C Example:**
```c
#include <math.h>

float with_nnan(float x, float y) 
    __attribute__((annotate("no-nans"))) {
    // Compiler knows x and y are never NaN
    if (isnan(x)) {  // Dead code - can be eliminated
        return 0.0f;
    }
    return fmaf(0.0f, x, y);  // Can optimize to just 'y'
}
```

**Command-line:**
```bash
clang -ffast-math ...        # Includes nnan
clang -ffinite-math-only ... # Includes nnan and ninf
```

---

#### ninf (NoInfs)
**Definition:** Fast-math flag asserting that **no infinity values** will occur.

**What it enables:**
```c
x * 1.0 → x       // Safe if x is never Inf
x / x → 1.0       // Safe if x is never Inf or 0
fma(0, x, y) → y  // Combined with nnan
```

**Example:**
```llvm
%div = fdiv ninf nnan float %x, %x
; Can optimize to:
%result = float 1.0

; Without ninf+nnan:
; Must preserve because:
;   Inf / Inf = NaN (not 1.0)
;   0.0 / 0.0 = NaN (not 1.0)
```

---

#### nsz (NoSignedZeros)
**Definition:** Allows ignoring the **sign of zero** (-0.0 vs +0.0).

**What it enables:**
```c
x + 0.0 → x          // Even if x is -0.0
x - 0.0 → x          // Safe with nsz
-0.0 == +0.0         // Can assume true
```

**Example:**
```llvm
%add = fadd nsz float %x, 0.0
; Optimized to:
%result = %x

; Without nsz:
; Must preserve because:
;   -0.0 + 0.0 = +0.0 (sign changes)
;   +0.0 + 0.0 = +0.0
```

---

#### fast (All Fast-Math Flags)
**Definition:** Shorthand for enabling **all** fast-math optimizations.

**Equivalent to:** `nnan ninf nsz arcp contract afn reassoc`

**LLVM IR:**
```llvm
%result = fadd fast float %a, %b
; Equivalent to:
%result = fadd nnan ninf nsz arcp contract afn reassoc float %a, %b
```

**C/C++:**
```bash
clang -ffast-math input.c
# Enables maximum floating-point optimizations
# Trade-off: May violate IEEE-754 strict semantics
```

**When to use:**
- Performance-critical code
- Numerical stability not critical
- No need for bit-exact reproducibility
- Scientific computing, graphics, game engines

**When NOT to use:**
- Financial calculations
- Cryptography
- Safety-critical systems
- Unit testing floating-point algorithms

---

### Hardware Architecture Terms

#### FMA3 (Intel 3-operand FMA)
**Definition:** Intel's AVX2 FMA instruction set with **3-operand destructive encoding**.

**Key Characteristics:**
- Available since Intel Haswell (2013)
- 3 operands: destination, source1, source2
- Destructive: destination is also a source operand
- Three forms: 132, 213, 231 (operand permutations)

**Assembly Example:**
```asm
; Form 213: dst = src1 * src2 + src3
vfmadd213ss xmm0, xmm1, xmm2    ; xmm0 = xmm0 * xmm1 + xmm2

; Form 132: dst = src1 * src3 + src2
vfmadd132ss xmm0, xmm1, xmm2    ; xmm0 = xmm0 * xmm2 + xmm1

; Form 231: dst = src2 * src3 + src1
vfmadd231ss xmm0, xmm1, xmm2    ; xmm0 = xmm1 * xmm2 + xmm0

; Memory operand (third operand from memory)
vfmadd213ss xmm0, xmm1, [rdi]   ; xmm0 = xmm0 * xmm1 + [rdi]
```

**Form Selection Example:**
```c
float compute(float a, float b, float c) {
    return a * b + c;
}

// If 'a' dies after this operation, prefer form 213:
// vfmadd213ss xmm0{a}, xmm1{b}, xmm2{c}  → xmm0 = a*b + c

// If 'c' dies after this operation, prefer form 132:
// vfmadd132ss xmm2{c}, xmm0{a}, xmm1{b}  → xmm2 = a*b + c
```

---

#### FMA4 (AMD 4-operand FMA)
**Definition:** AMD's 4-operand FMA instruction set (non-destructive).

**Key Characteristics:**
- AMD Bulldozer/Piledriver only (2011-2012)
- 4 operands: destination + 3 independent sources
- Non-destructive: destination is separate
- Deprecated (AMD now uses FMA3)

**Assembly Example:**
```asm
; FMA4: dst = src1 * src2 + src3 (all independent)
vfmaddss xmm0, xmm1, xmm2, xmm3   ; xmm0 = xmm1 * xmm2 + xmm3
vfmaddss xmm0, xmm1, [rdi], xmm3  ; xmm0 = xmm1 * [rdi] + xmm3
vfmaddss xmm0, xmm1, xmm2, [rdi]  ; xmm0 = xmm1 * xmm2 + [rdi]
```

---

### Rounding and Precision Terms

#### Single Rounding
**Definition:** Rounding occurs **once** at the final result.

**Computation Steps:**
1. Compute `a × b` with infinite precision (conceptually)
2. Add `c` to the exact product (still infinite precision)
3. Round final result to target format (e.g., float32)

**Example (float32):**
```
a = 1.0000001 (slightly above 1.0)
b = 1.0000001
c = -0.0000001

Step 1: a × b = 1.0000002000001 (exact, many bits)
Step 2: + c  = 1.0000001000001 (exact, many bits)
Step 3: Round to float32 → 1.0000001 (single rounding)
```

---

#### Double Rounding
**Definition:** Rounding occurs **twice** - once for multiply, once for add.

**Computation Steps:**
1. Compute `a × b` → Round to target format
2. Take rounded product, add `c` → Round to target format

**Example (float32):**
```
a = 1.0000001
b = 1.0000001  
c = -0.0000001

Step 1: a × b = 1.0000002000001 → Round → 1.0000002 (first rounding)
Step 2: 1.0000002 + c = 1.0000001 → Round → 1.0000001 (second rounding)

Note: Result differs from single rounding in edge cases!
```

**When Double Rounding Causes Issues:**
```python
# Python example showing double rounding error
from decimal import Decimal, getcontext
getcontext().prec = 10

# Values carefully chosen to show rounding difference
a = Decimal('1.000000001')
b = Decimal('1.000000001')
c = Decimal('-0.000000001')

# Single rounding (simulated)
exact = a * b + c
single_round = float(exact)

# Double rounding (actual behavior without FMA)
first_round = float(a * b)
double_round = float(first_round + float(c))

print(f"Single: {single_round}")
print(f"Double: {double_round}")
# Results may differ!
```

---

#### Contraction
**Definition:** Combining separate multiply and add operations into a single FMA operation.

**Pattern:**
```
Before: (fmul a, b) then (fadd result, c)
After:  (fma a, b, c)
```

**Benefits:**
1. **Performance:** One instruction instead of two
2. **Accuracy:** Single rounding instead of double rounding
3. **Register pressure:** Eliminates intermediate value
4. **Latency:** Shorter dependency chains

**Example Trace:**
```llvm
; Before contraction:
define float @before(float %a, float %b, float %c) {
entry:
  %mul = fmul contract float %a, %b    ; 4 cycle latency
  %add = fadd contract float %mul, %c  ; 3 cycle latency (depends on %mul)
  ret float %add
  ; Total latency: 4 + 3 = 7 cycles
  ; Instructions: 2 (fmul, fadd)
  ; Registers: 4 (%a, %b, %c, %mul)
}

; After contraction:
define float @after(float %a, float %b, float %c) {
entry:
  %fma = call float @llvm.fma.f32(float %a, float %b, float %c)  ; 5 cycle latency
  ret float %fma
  ; Total latency: 5 cycles (better!)
  ; Instructions: 1 (fma)
  ; Registers: 3 (%a, %b, %c)
}
```

---
