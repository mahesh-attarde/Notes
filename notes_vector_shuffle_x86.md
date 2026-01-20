# X86 Vector Shuffle 
## Overview

Vector shuffles rearrange elements within SIMD vectors.  The X86 architecture provides numerous shuffle instructions optimized for different patterns: 

| Category | Instructions | Best For |
|----------|-------------|----------|
| Byte-level | PSHUFB | Arbitrary byte permutation |
| Word/Dword | PSHUFD, PSHUFHW, PSHUFLW | Element broadcast, rotation |
| Blending | BLEND, BLENDV | Merging two vectors |
| Interleaving | UNPACK | AoS↔SoA, widening |
| Cross-lane | VPERM, VPERM2X128 | Full permutation |
| Broadcast | VBROADCAST | Scalar splat |

---

## Shuffle Operations

### 1. PSHUFB - Packed Shuffle Bytes

**Full Name**:  Packed Shuffle Bytes

| Property | Value |
|----------|-------|
| **Instructions** | `PSHUFB`, `VPSHUFB` |
| **ISA** | SSSE3, AVX, AVX2, AVX-512BW |
| **Element Size** | 8-bit (bytes) |
| **Vector Sizes** | 128-bit, 256-bit, 512-bit |
| **Latency** | 1 cycle |
| **Throughput** | 1 per cycle |

**Description**:  
Performs arbitrary byte-level permutation within 128-bit lanes using a control mask vector. Each byte in the mask selects which source byte appears in the result.  Setting bit 7 of a mask byte zeros that output position. 

**LLVM Node**: `X86ISD::PSHUFB`

```cpp
/// Shuffle 16 8-bit values within a vector. 
PSHUFB,
```

**Operation**:
```
for i in 0..15:
    if mask[i] & 0x80:
        result[i] = 0
    else:
        result[i] = src[mask[i] & 0x0F]
```

**Use Cases**:
- Lookup tables (nibble-to-hex conversion)
- Byte reversal / endian swap
- Arbitrary byte rearrangement
- Bitwise operations via lookup
- String processing

**Example**:
```llvm
; Reverse bytes in each 128-bit lane
%mask = <16 x i8> <i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8,
                   i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>
%result = call <16 x i8> @llvm.x86.ssse3.pshuf.b. 128(<16 x i8> %src, <16 x i8> %mask)
```

**Performance Notes**:
-  Very fast (1 cycle latency)
- Requires mask in register/memory (not immediate)
- Lane-constrained:  cannot cross 128-bit boundaries in 256/512-bit vectors

---

### 2. PSHUFD/PSHUFHW/PSHUFLW - Word/Dword Shuffles

**Full Name**:  Packed Shuffle Doublewords / High Words / Low Words

| Property | PSHUFD | PSHUFHW | PSHUFLW |
|----------|--------|---------|---------|
| **Element Size** | 32-bit | 16-bit (high) | 16-bit (low) |
| **ISA** | SSE2+ | SSE2+ | SSE2+ |
| **Control** | 8-bit immediate | 8-bit immediate | 8-bit immediate |
| **Latency** | 1 cycle | 1 cycle | 1 cycle |

**Description**:  
Immediate-controlled shuffles for 32-bit or 16-bit elements.  PSHUFD shuffles all four dwords; PSHUFHW/PSHUFLW shuffle only the high/low four words while leaving the other half unchanged.

**LLVM Nodes**:  `X86ISD::PSHUFD`, `X86ISD::PSHUFHW`, `X86ISD::PSHUFLW`

**Immediate Encoding (PSHUFD)**:
```
imm[1:0] → selects source for element 0
imm[3:2] → selects source for element 1
imm[5:4] → selects source for element 2
imm[7:6] → selects source for element 3
```

**Use Cases**:
- Broadcasting single element to all positions
- Vector rotation
- Preparing operands for horizontal operations
- Data rearrangement

**Common Patterns**:
| Pattern | Immediate | Result |
|---------|-----------|--------|
| Broadcast elem 0 | `0x00` | `[a0,a0,a0,a0]` |
| Broadcast elem 3 | `0xFF` | `[a3,a3,a3,a3]` |
| Reverse | `0x1B` | `[a3,a2,a1,a0]` |
| Rotate left | `0x39` | `[a1,a2,a3,a0]` |

**Example**:
```llvm
; Broadcast element 0 to all positions
%result = shufflevector <4 x i32> %v, <4 x i32> undef,
                        <4 x i32> <i32 0, i32 0, i32 0, i32 0>
; Generates: pshufd $0x00, %xmm0, %xmm0
```

**Performance Notes**:
-  Fastest shuffle type (immediate-controlled)
- No additional register needed for mask
- Lane-constrained in AVX/AVX-512 variants

---

### 3. BLEND - Element Selection

**Full Name**:  Blend Packed Values

| Variant | Element Size | Control | ISA |
|---------|-------------|---------|-----|
| `PBLENDW` | 16-bit | Immediate | SSE4.1 |
| `BLENDPS` | 32-bit float | Immediate | SSE4.1 |
| `BLENDPD` | 64-bit double | Immediate | SSE4.1 |
| `PBLENDVB` | 8-bit | Variable (XMM0) | SSE4.1 |
| `BLENDVPS` | 32-bit float | Variable (XMM0) | SSE4.1 |
| `BLENDVPD` | 64-bit double | Variable (XMM0) | SSE4.1 |
| `VBLENDPS` | 32-bit float | Immediate | AVX |
| `VPBLENDMD` | 32-bit | Mask register | AVX-512 |

**Description**:  
Selects elements from two source vectors based on an immediate mask or variable condition. Immediate blends use a bit mask; variable blends use the sign bit of each element in the mask vector.

**LLVM Nodes**: `X86ISD::BLENDI`, `X86ISD::BLENDV`

```cpp
/// Blend where the selector is an immediate. 
BLENDI,

/// Dynamic (non-constant condition) vector blend where only the sign bits
/// of the condition elements are used.
/// Operands are in VSELECT order:  MASK, TRUE, FALSE
BLENDV,
```

**Operation (Immediate Blend)**:
```
for i in 0..N: 
    result[i] = (imm >> i) & 1 ? src2[i] : src1[i]
```

**Use Cases**:
- Conditional element selection
- Merging partial results
- Implementing vector select/ternary
- Masked operations

**Example**:
```llvm
; Select:  result = {a0, b1, a2, b3}
%result = shufflevector <4 x float> %a, <4 x float> %b,
                        <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; Generates:  blendps $0x0A, %xmm1, %xmm0
```

**Performance Notes**:
-  Immediate blends:  1 cycle latency
- Variable blends: 1-2 cycles, require mask setup
- AVX-512 mask blends are very efficient

---

### 4. UNPACK - Interleave Operations

**Full Name**: Unpack and Interleave

| Variant | Operation | Element Sizes |
|---------|-----------|---------------|
| `PUNPCKLBW` | Unpack low bytes | 8-bit |
| `PUNPCKHBW` | Unpack high bytes | 8-bit |
| `PUNPCKLWD` | Unpack low words | 16-bit |
| `PUNPCKHWD` | Unpack high words | 16-bit |
| `PUNPCKLDQ` | Unpack low dwords | 32-bit |
| `PUNPCKHDQ` | Unpack high dwords | 32-bit |
| `PUNPCKLQDQ` | Unpack low qwords | 64-bit |
| `PUNPCKHQDQ` | Unpack high qwords | 64-bit |
| `UNPCKLPS` | Unpack low singles | 32-bit float |
| `UNPCKHPS` | Unpack high singles | 32-bit float |
| `UNPCKLPD` | Unpack low doubles | 64-bit double |
| `UNPCKHPD` | Unpack high doubles | 64-bit double |

**Description**:  
Interleaves elements from the low (UNPCKL) or high (UNPCKH) halves of two vectors. Essential for transposition and AoS/SoA conversions.

**LLVM Nodes**: `X86ISD::UNPCKL`, `X86ISD::UNPCKH`

**Operation (UNPCKLPS - 32-bit elements)**:
```
A:  [a3, a2, a1, a0]
B: [b3, b2, b1, b0]
UNPCKLPS Result: [b1, a1, b0, a0]
UNPCKHPS Result: [b3, a3, b2, a2]
```

**Visual Diagram**:
```
UNPCKL (low elements):
  A:  [ A3 | A2 | A1 | A0 ]
  B:  [ B3 | B2 | B1 | B0 ]
       ↓         ↓    ↓
  R:  [ B1 | A1 | B0 | A0 ]

UNPCKH (high elements):
  A: [ A3 | A2 | A1 | A0 ]
  B: [ B3 | B2 | B1 | B0 ]
   ↓    ↓
  R: [ B3 | A3 | B2 | A2 ]
```

**Use Cases**:
- Matrix transpose
- AoS (Array of Structures) ↔ SoA (Structure of Arrays) conversion
- Widening operations (with zero vector)
- Complex number operations
- Pixel format conversion

**Example - 4x4 Matrix Transpose**: 
```llvm
; Step 1: Unpack pairs
%t0 = shufflevector <4 x float> %row0, <4 x float> %row1,
                    <4 x i32> <i32 0, i32 4, i32 1, i32 5>  ; unpcklps
%t1 = shufflevector <4 x float> %row0, <4 x float> %row1,
                    <4 x i32> <i32 2, i32 6, i32 3, i32 7>  ; unpckhps
%t2 = shufflevector <4 x float> %row2, <4 x float> %row3,
                    <4 x i32> <i32 0, i32 4, i32 1, i32 5>  ; unpcklps
%t3 = shufflevector <4 x float> %row2, <4 x float> %row3,
                    <4 x i32> <i32 2, i32 6, i32 3, i32 7>  ; unpckhps

; Step 2: Final interleave
%col0 = shufflevector <4 x float> %t0, <4 x float> %t2,
                      <4 x i32> <i32 0, i32 1, i32 4, i32 5>
%col1 = shufflevector <4 x float> %t0, <4 x float> %t2,
                      <4 x i32> <i32 2, i32 3, i32 6, i32 7>
%col2 = shufflevector <4 x float> %t1, <4 x float> %t3,
                      <4 x i32> <i32 0, i32 1, i32 4, i32 5>
%col3 = shufflevector <4 x float> %t1, <4 x float> %t3,
                      <4 x i32> <i32 2, i32 3, i32 6, i32 7>
```

**Performance Notes**:
-  1 cycle latency
- Very efficient for structured interleaving
- Lane-constrained in AVX (operates within 128-bit lanes)

---

### 5. VPERM - Variable Permute

**Full Name**: Variable Permute

| Instruction | Element Size | Sources | ISA |
|-------------|-------------|---------|-----|
| `VPERMD` | 32-bit int | 1 | AVX2 |
| `VPERMPS` | 32-bit float | 1 | AVX2 |
| `VPERMQ` | 64-bit int | 1 (imm) | AVX2 |
| `VPERMPD` | 64-bit double | 1 (imm) | AVX2 |
| `VPERMI2D` | 32-bit | 2 | AVX-512 |
| `VPERMT2D` | 32-bit | 2 | AVX-512 |
| `VPERMI2PS` | 32-bit float | 2 | AVX-512 |
| `VPERMT2PS` | 32-bit float | 2 | AVX-512 |

**Description**:  
Full cross-lane permutation using an index vector. Unlike lane-constrained shuffles, VPERM can move any element to any position across the entire vector.

**LLVM Nodes**: `X86ISD::VPERMV`, `X86ISD::VPERMV3`, `X86ISD::VPERMI`

```cpp
// Variable Permute (VPERM).
// Res = VPERMV MaskV, V0
VPERMV,

// 3-op Variable Permute (VPERMT2).
// Res = VPERMV3 V0, MaskV, V1
VPERMV3,
```

**Operation (VPERMD)**:
```
for i in 0..7:
    result[i] = src[index[i] & 0x7]
```

**Use Cases**:
- Arbitrary element reordering
- Gather-like operations
- LUT-based computations
- Sorting networks
- Histogram computation

**Example**:
```llvm
; Reverse all 8 elements (cross-lane)
%indices = <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
%result = call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %src, <8 x i32> %indices)
```

**Performance Notes**:
- 3 cycle latency (cross-lane penalty)
- More flexible than in-lane shuffles
- AVX-512 two-source permutes are powerful for merge operations

---

### 6. VPERM2X128/SHUF128 - Lane Permute

**Full Name**:  Permute 128-bit Lanes

| Instruction | Vector Size | ISA |
|-------------|-------------|-----|
| `VPERM2F128` | 256-bit float | AVX |
| `VPERM2I128` | 256-bit int | AVX2 |
| `VSHUF32X4` | 512-bit (128-bit granularity) | AVX-512 |
| `VSHUF64X2` | 512-bit (128-bit granularity) | AVX-512 |

**Description**:  
Shuffles entire 128-bit lanes between two 256-bit or 512-bit vectors. Can also zero lanes. 

**LLVM Nodes**:  `X86ISD::VPERM2X128`, `X86ISD::SHUF128`

**Immediate Encoding (VPERM2X128)**:
```
Bits [1:0]:  Source for result[127:0]
  00 = src1[127:0]
  01 = src1[255:128]
  10 = src2[127:0]
  11 = src2[255:128]
Bit [3]:  Zero result[127:0] if set
Bits [5:4]: Source for result[255:128]
Bit [7]: Zero result[255:128] if set
```

**Use Cases**:
- Lane swapping
- Lane broadcast
- Combining lane halves from different vectors
- Setting up for cross-lane operations

**Example**: 
```llvm
; Swap 128-bit lanes
%swapped = shufflevector <4 x double> %v, <4 x double> undef,
                         <4 x i32> <i32 2, i32 3, i32 0, i32 1>
; Generates: vperm2f128 $0x01, %ymm0, %ymm0, %ymm0

; Broadcast low lane to both lanes
%broadcast = shufflevector <4 x double> %v, <4 x double> undef,
                           <4 x i32> <i32 0, i32 1, i32 0, i32 1>
; Generates: vperm2f128 $0x00, %ymm0, %ymm0, %ymm0
```

**Performance Notes**:
- 3 cycle latency
- Essential for 256/512-bit cross-lane work
- Can be avoided if algorithm stays within lanes

---

### 7. SHUFP - Shuffle Packed FP

**Full Name**: Shuffle Packed Single/Double-Precision

| Instruction | Element Size | ISA |
|-------------|-------------|-----|
| `SHUFPS` | 32-bit float | SSE |
| `SHUFPD` | 64-bit double | SSE2 |
| `VSHUFPS` | 32-bit float | AVX |
| `VSHUFPD` | 64-bit double | AVX |

**Description**:  
Two-source shuffle using immediate control. For SHUFPS, low two elements come from first source, high two from second source.

**LLVM Node**: `X86ISD::SHUFP`

**Operation (SHUFPS)**:
```
result[0] = src1[imm[1:0]]
result[1] = src1[imm[3:2]]
result[2] = src2[imm[5:4]]
result[3] = src2[imm[7:6]]
```

**Use Cases**:
- Combining elements from two vectors
- Building specific patterns from two sources
- FP data rearrangement

**Example**:
```llvm
; Take low half from A, high half from B
%result = shufflevector <4 x float> %a, <4 x float> %b,
                        <4 x i32> <i32 0, i32 1, i32 4, i32 5>
; Generates: shufps $0x44, %xmm1, %xmm0
```

**Performance Notes**:
-  1 cycle latency
- Immediate-controlled (no mask register)
- Lane-constrained in AVX

---

### 8. VPERMILP - In-Lane Permute

**Full Name**:  Permute In-Lane Single/Double-Precision

| Instruction | Element Size | Control | ISA |
|-------------|-------------|---------|-----|
| `VPERMILPS` | 32-bit float | Immediate or vector | AVX |
| `VPERMILPD` | 64-bit double | Immediate or vector | AVX |

**Description**:  
Permutes floating-point elements within each 128-bit lane independently. Does not cross lane boundaries.

**LLVM Nodes**: `X86ISD::VPERMILPV`, `X86ISD::VPERMILPI`

```cpp
VPERMILPV,  // Variable permute in-lane (index from vector)
VPERMILPI,  // Immediate permute in-lane
```

**Operation (per 128-bit lane)**:
```
for i in 0..3:  // VPERMILPS
    lane_result[i] = lane_src[index[i] & 0x3]
```

**Use Cases**:
- Lane-local shuffles
- Broadcast within lanes
- Setting up for horizontal operations
- Duplicating elements within lanes

**Example**: 
```llvm
; Reverse elements within each 128-bit lane
%result = shufflevector <8 x float> %v, <8 x float> undef,
                        <8 x i32> <i32 3, i32 2, i32 1, i32 0,
                                   i32 7, i32 6, i32 5, i32 4>
; Generates: vpermilps $0x1B, %ymm0, %ymm0
```

**Performance Notes**:
-  1 cycle latency
- No cross-lane penalty
- Preferred over cross-lane permutes when possible

---

### 9. PALIGNR/VALIGN - Byte Alignment

**Full Name**: Packed Align Right / Vector Align

| Instruction | Granularity | ISA |
|-------------|------------|-----|
| `PALIGNR` | Byte | SSSE3 |
| `VPALIGNR` | Byte (per lane) | AVX |
| `VALIGND` | Dword | AVX-512 |
| `VALIGNQ` | Qword | AVX-512 |

**Description**:  
Concatenates two vectors and extracts a shifted portion.  PALIGNR concatenates, shifts right by immediate bytes, and takes the low portion. 

**LLVM Nodes**:  `X86ISD::PALIGNR`, `X86ISD::VALIGN`

```cpp
PALIGNR,  // Intra-lane alignr (SSE/AVX)
VALIGN,   // AVX512 inter-lane alignr
```

**Operation (PALIGNR)**:
```
temp[255:0] = concat(src2[127:0], src1[127:0])  // 256-bit temporary
result = temp >> (imm * 8)                       // Shift right by imm bytes
result = result[127:0]                           // Take low 128 bits
```

**Use Cases**:
- Byte rotation
- Sliding window operations
- Misaligned data handling
- String comparison
- Shift across element boundaries

**Example**:
```llvm
; Rotate vector left by 4 bytes
%result = call <16 x i8> @llvm.x86.ssse3.palign.r. 128(
    <16 x i8> %a, <16 x i8> %a, i8 12)
; Concatenates a: a, shifts right by 12 bytes = rotate left by 4
```

**Performance Notes**:
-  1 cycle latency
- Lane-constrained in AVX (VPALIGNR)
- AVX-512 VALIGN can cross lanes

---

### 10. VBROADCAST - Splat Operations

**Full Name**: Broadcast

| Instruction | Source | Destination | ISA |
|-------------|--------|-------------|-----|
| `VBROADCASTSS` | 32-bit mem/xmm | xmm/ymm/zmm | AVX |
| `VBROADCASTSD` | 64-bit mem/xmm | ymm/zmm | AVX |
| `VBROADCASTF128` | 128-bit mem | ymm | AVX |
| `VPBROADCASTB` | 8-bit | xmm/ymm/zmm | AVX2 |
| `VPBROADCASTW` | 16-bit | xmm/ymm/zmm | AVX2 |
| `VPBROADCASTD` | 32-bit | xmm/ymm/zmm | AVX2 |
| `VPBROADCASTQ` | 64-bit | xmm/ymm/zmm | AVX2 |

**Description**:  
Replicates a scalar or small vector to fill an entire vector register. Very efficient, especially when combined with memory operands.

**LLVM Node**: `X86ISD::VBROADCAST`, `X86ISD::VBROADCASTM`

```cpp
/// Broadcast (splat) scalar or element 0 of a vector.  If the operand is
/// a vector, this node may change the vector length as part of the splat. 
VBROADCAST,
/// Broadcast mask to vector.
VBROADCASTM,
```

**Use Cases**:
- Constant replication
- Scalar-vector operations
- FMA with scalar multiplier
- Initializing vectors

**Example**: 
```llvm
; Broadcast scalar to all elements
%scalar = load float, ptr %ptr
%vec = insertelement <8 x float> undef, float %scalar, i32 0
%broadcast = shufflevector <8 x float> %vec, <8 x float> undef,
                           <8 x i32> zeroinitializer
; Generates: vbroadcastss (%rdi), %ymm0
```

**Performance Notes**:
-  1 cycle latency (often fused with load)
- Memory broadcast eliminates explicit load
- AVX-512 embedded broadcast reduces code size

---

### 11. MOV Variants - Special Shuffles

**Full Name**: Move with Shuffle Semantics

| Instruction | Operation | Result Pattern |
|-------------|-----------|----------------|
| `MOVDDUP` | Duplicate low qword | `[a0, a0]` |
| `MOVSHDUP` | Duplicate odd elements | `[a3, a3, a1, a1]` |
| `MOVSLDUP` | Duplicate even elements | `[a2, a2, a0, a0]` |
| `MOVLHPS` | Move low to high | `[b1, b0, a1, a0]` |
| `MOVHLPS` | Move high to low | `[a3, a2, b3, b2]` |
| `MOVSD` | Move scalar double | Scalar insert |
| `MOVSS` | Move scalar single | Scalar insert |

**LLVM Nodes**:
```cpp
MOVDDUP,
MOVSHDUP,
MOVSLDUP,
MOVLHPS,
MOVHLPS,
MOVSD,
MOVSS,
MOVSH,
```

**Use Cases**:
- Specific duplication patterns
- Setting up for horizontal ops
- Complex number operations
- Scalar insertion

**Example**:
```llvm
; Duplicate low element
%dup = shufflevector <2 x double> %v, <2 x double> undef,
                     <2 x i32> <i32 0, i32 0>
; Generates: movddup %xmm0, %xmm0
```

**Performance Notes**:
-  1 cycle latency
- Often more efficient than general shuffle for specific patterns

---

### 12. PACK - Narrowing Operations

**Full Name**: Pack with Saturation

| Instruction | From | To | Saturation |
|-------------|------|-----|-----------|
| `PACKSSWB` | 16-bit | 8-bit | Signed |
| `PACKSSDW` | 32-bit | 16-bit | Signed |
| `PACKUSWB` | 16-bit | 8-bit | Unsigned |
| `PACKUSDW` | 32-bit | 16-bit | Unsigned |

**Description**:  
Narrows elements from two source vectors into a single destination vector with saturation.

**LLVM Nodes**: `X86ISD::PACKSS`, `X86ISD::PACKUS`

```cpp
// Saturated signed/unsigned packing. 
PACKSS,
PACKUS,
```

**Operation (PACKSSWB)**:
```
A:  [a7, a6, a5, a4, a3, a2, a1, a0]  (8 x i16)
B: [b7, b6, b5, b4, b3, b2, b1, b0]  (8 x i16)
Result: [sat(b7).. sat(b0), sat(a7)..sat(a0)]  (16 x i8)
```

**Use Cases**:
- Type conversion with saturation
- Image processing (clamp to byte range)
- Audio processing
- Compression

**Performance Notes**:
-  1 cycle latency
- Combines narrowing and saturation efficiently

---

## Performance Summary

### Latency and Throughput by Category

| Category | Instructions | Latency | Throughput | Port |
|----------|-------------|---------|------------|------|
| **In-lane Immediate** | PSHUFD, SHUFPS, BLENDI | 1 cycle | 1/cycle | p5 |
| **In-lane Variable** | PSHUFB, VPERMILP | 1 cycle | 1/cycle | p5 |
| **Broadcast** | VBROADCAST* | 1 cycle | 0.5/cycle | p0,p5 |
| **Cross-lane Immediate** | VPERM2X128 | 3 cycles | 1/cycle | p5 |
| **Cross-lane Variable** | VPERMD, VPERMPS | 3 cycles | 1/cycle | p5 |
| **Two-source Permute** | VPERMT2D | 3 cycles | 1/cycle | p5 |
| **Blend Variable** | BLENDV* | 2 cycles | 1/cycle | p0,p1 |
| **Unpack** | UNPCK* | 1 cycle | 1/cycle | p5 |
| **Pack** | PACK* | 1 cycle | 1/cycle | p5 |

### Performance Tiers

```
 Fastest (1 cycle, no cross-lane):
   PSHUFD, PSHUFHW, PSHUFLW, SHUFPS, SHUFPD
   BLENDI, PSHUFB, VPERMILP
   UNPACK*, PACK*, PALIGNR
   MOVDDUP, MOVSHDUP, MOVSLDUP
   VBROADCAST*

 Moderate (1-3 cycles, may cross lanes):
   BLENDV*, VPERM2X128, SHUF128

 Cross-lane (3+ cycles):
   VPERMD, VPERMPS, VPERMQ, VPERMPD
   VPERMT2*, VPERMI2*
```

### Optimization Guidelines

1. **Prefer in-lane operations** when possible
2. **Use immediate-controlled shuffles** over variable when mask is constant
3. **Combine with memory operands** to reduce instruction count
4. **Use broadcast loads** instead of load + shuffle
5. **Consider algorithm redesign** to avoid cross-lane shuffles

---

## Use Case Reference

| Task | Recommended Instructions | Notes |
|------|-------------------------|-------|
| **Broadcast scalar** | `VBROADCAST*` | Fuses with load |
| **Element permutation (in-lane)** | `PSHUFD`, `VPERMILP` | 1 cycle |
| **Element permutation (cross-lane)** | `VPERMD`, `VPERMPS` | 3 cycles |
| **Interleave vectors** | `UNPCKL`, `UNPCKH` | AoS↔SoA |
| **Matrix transpose** | `UNPACK` + `SHUFPS` | See example above |
| **Conditional merge** | `BLEND*` | Immediate faster |
| **Byte-level rearrange** | `PSHUFB` | Needs mask register |
| **Lane swap** | `VPERM2X128` | 256-bit vectors |
| **Rotate bytes** | `PALIGNR` | Within lanes |
| **Zero extension** | `UNPCKL` with zero | Widening |
| **Narrowing** | `PACK*` | With saturation |
| **Lookup table** | `PSHUFB` | 16-entry LUT |
| **Horizontal reduction** | `HADD` + shuffles | Sum/product |

---

## Source Code References

### LLVM Source Files

| File | Description |
|------|-------------|
| [`X86ISelLowering.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.h) | X86ISD opcode definitions |
| [`X86ISelLowering.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLowering.cpp) | Shuffle lowering implementation |
| [`X86ShuffleDecode.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/MCTargetDesc/X86ShuffleDecode.h) | Mask decode function declarations |
| [`X86ShuffleDecode.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/MCTargetDesc/X86ShuffleDecode. cpp) | Mask decode implementations |
| [`X86InstCombineIntrinsic.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstCombineIntrinsic.cpp) | Shuffle simplification |

### Test Files

| Directory | Content |
|-----------|---------|
| [`vector-shuffle-128-*. ll`](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/X86) | 128-bit shuffle tests |
| [`vector-shuffle-256-*.ll`](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/X86) | 256-bit shuffle tests |
| [`vector-shuffle-512-*.ll`](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/X86) | 512-bit shuffle tests |
| [`avx512-shuffles/`](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/X86/avx512-shuffles) | AVX-512 specific tests |

