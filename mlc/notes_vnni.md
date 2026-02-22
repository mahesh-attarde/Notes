# AVX VNNI & AVX-512 VNNI
## 1) What is VNNI?
Vector Neural Network Instructions, not limited to neural networks. It accelerates a very common inner loop found in:
- GEMM / MatMul (matrix multiplication)
- Convolution (direct or lowered forms)
- Dot products over many small integers
- Inference workloads using INT8 or BF16
- Signal processing / feature transforms

Workload Character
computation is a **dot product / multiply-accumulate** over small integers:
- acc += sum over k of (a[k] * b[k])
VNNI accelerates this by **fusing** steps that otherwise take multiple instructions.
TODO: Why Fusing? 

## 2) The problem VNNI solves
### 2.1 Without VNNI
In INT8 inference you often multiply 8-bit integers and accumulate into 32-bit integers:
- Inputs: int8 or uint8
- Accumulator: int32

Without VNNI, a typical SIMD approach needs multiple steps:
1. widen int8/uint8 to int16 or int32
2. multiply
3. add into int32 accumulator

That costs extra instructions, bandwidth, and registers.

### 2.2 What VNNI does
VNNI provides instructions that conceptually do, in one go:

- multiply packed 8-bit elements
- horizontally sum partial products in groups
- accumulate into 32-bit lanes

This increases throughput and usually reduces instruction count.

## 3) AVX VNNI vs AVX-512 VNNI (high-level)

### 3.1 AVX VNNI
- Operates on 256-bit YMM registers (AVX2 width).
- Adds VNNI-style dot-product instructions without requiring AVX-512.
- Useful on systems where AVX-512 is unavailable but AVX2 exists.
### 3.2 AVX-512 VNNI
- Operates on 512-bit ZMM registers (AVX-512 width).
- Adds masking (k registers), more lanes, and potentially higher peak throughput.
- Often used alongside other AVX-512 capabilities (masking, more registers, etc.).

Rule of thumb: AVX-512 VNNI can offer ~2x vector width vs 256-bit VNNI, but real speedup depends on frequency behavior, memory, and whether you are compute-bound.

## 4) What operations VNNI accelerates (INT8 path)

The canonical VNNI operation is a dot product of 8-bit values accumulating into 32-bit:

Common operand type pairings:
1. signed x signed: int8 * int8
2. unsigned x signed: uint8 * int8 (common: activations unsigned, weights signed)

Accumulator is typically int32.

## 5) Core instruction families and semantics

### 5.1 AVX-512 VNNI: VPDPBUSD and VPDPBUSDS
These are among the most common AVX-512 VNNI instructions.
#### VPDPBUSD (Unsigned bytes x Signed bytes -> Dword accumulate)
Think of the vector as bytes grouped into chunks of 4 bytes per 32-bit output lane.
For each 32-bit lane i:
- dst[i] = acc[i] + (u8(a[4*i+0]) * s8(b[4*i+0])
                  + u8(a[4*i+1]) * s8(b[4*i+1])
                  + u8(a[4*i+2]) * s8(b[4*i+2])
                  + u8(a[4*i+3]) * s8(b[4*i+3]))

Notes:
- u8(x) means treat the byte as unsigned (0..255)
- s8(x) means treat the byte as signed (-128..127)
- The four products are summed to an int32 and added to the accumulator.

#### VPDPBUSDS (same multiply, but saturating accumulate)

Semantically similar, but the final accumulate is **signed saturating** to the int32 range. This is less common in standard GEMM paths that want full int32 accumulation headroom, but it exists for specific quantization pipelines.

Masking: AVX-512 forms can use a mask register k to enable/disable lanes.

### 5.2 AVX VNNI (256-bit): similar dot-product accumulate

AVX VNNI introduces dot-product accumulate operations for 256-bit vectors. You usually access them via intrinsics that map to VNNI instructions.

The semantic idea is the same:
- bytes x bytes
- 4 products per 32-bit lane
- accumulate into int32

## 6) Understanding the “4 bytes per dword lane” grouping

VNNI groups operands like this:

- output dword lane 0 uses bytes 0..3
- output dword lane 1 uses bytes 4..7
- etc.

So each 32-bit output lane is one dot product of 4 byte-pairs.

Why it matters:
- data layout / packing must match the grouping
- kernel design depends on how you feed these groups efficiently
- avoiding shuffles is critical for performance


## 7) Example: what one VNNI lane computes

For one 32-bit lane:

- a = [a0, a1, a2, a3] treated as unsigned bytes
- b = [b0, b1, b2, b3] treated as signed bytes
- acc is int32

Then:

- out = acc + (a0*b0) + (a1*b1) + (a2*b2) + (a3*b3)

This happens across:
- 8 lanes for a 256-bit vector of int32 (8 x 32-bit lanes)
- 16 lanes for a 512-bit vector of int32 (16 x 32-bit lanes)


## 8) Typical usage in GEMM / MatMul (INT8 inference)

A typical INT8 GEMM computes:

- C[m,n] = sum over k of (A[m,k] * B[k,n])

Often:
- A is uint8 activations
- B is int8 weights
- C accumulates in int32
- later: scaling + bias + activation, then store to int8/uint8/fp16/fp32

VNNI helps because the k-loop becomes repeated dot-product-accumulate operations on packed blocks.

### 8.1 Packing / layout matters

To use VNNI efficiently you typically:
- pack B (weights) into cache-friendly panels
- arrange A blocks so that groups of 4 bytes align with the instruction’s grouping
- choose micro-kernel shapes that match register availability

---

GCC/Clang commonly use:
- AVX-512 VNNI: `-mavx512vnni -mavx512bw -mavx512f`
- AVX VNNI: `-mavx2 -mavxvnni`


## 9) Performance characteristics and practical tips

### 9.1 VNNI is compute-dense

Each instruction performs multiple multiplies and adds, so:
- kernels may be compute-bound when data is in L1/L2
- blocking and prefetching are important

### 9.2 Watch AVX-512 frequency behavior

On some CPUs, heavy AVX-512 use can reduce turbo frequency. Net outcome:
- AVX-512 VNNI may or may not outperform AVX VNNI depending on platform and workload size.

### 11.3 Avoid extra shuffles

If your data is not already packed correctly, you may lose performance to:
- byte shuffles (vpshufb)
- permutes (vperm*)
- unpack/pack overhead

Good kernels choose layouts so operands can be loaded directly with minimal rearrangement.


## 10) Relationship to BF16 (and “VNNI” naming)

Intel also provides BF16-related acceleration under AVX-512 BF16 (separate feature). The theme is similar: fused operations for dot products / accumulation on inference/training-friendly formats.

If BF16 matters to you:
- learn AVX-512 BF16 as a separate capability
- learn conversions (float32 <-> bfloat16) and how they feed compute kernels

## 11) When should you use AVX VNNI vs AVX-512 VNNI?

Choose AVX VNNI when:
- you want broad compatibility on AVX2-era machines
- you want INT8 speedups without AVX-512
- you want less risk of AVX-512 frequency downclock

Choose AVX-512 VNNI when:
- you control deployment hardware (server fleet)
- you can benefit from 512-bit width + masking + more registers
- you have tuned kernels and the platform’s frequency behavior is acceptable

Best practice: implement both and select via runtime dispatch.

- VNNI = dot-product-accumulate for INT8/BF16-like workloads
- Main benefit = fewer instructions + higher throughput in inner loops
- Data layout is critical
- AVX VNNI = 256-bit dot-product accumulate
- AVX-512 VNNI = 512-bit dot-product accumulate + masking


## KERNELS

1.GEMM: INT8/UINT8 VNNI
```cpp
// C[0..15] += sum_{k=0..K-1} (uint8_t)A[k] * (int8_t)B[k][j]
void gemm_u8s8s32_1x16_vnni(
    const uint8_t* A,           // [K]
    const int8_t*  B,           // [K][16] contiguous, K-major
    int32_t*       C,           // [16]
    int K)
{
    __m512i acc0 = _mm512_loadu_si512((const void*)C); // 16x s32
    // VNNI dpbusd consumes 4 bytes of A and 4 bytes per lane of B per step.
    // Here we broadcast 4 A bytes to all lanes via a 32-bit value replicated.
    for (int k = 0; k < K; k += 4) {
        uint32_t a4;
        // Load 4 u8 safely (unaligned ok)
        a4  = (uint32_t)A[k + 0];
        a4 |= (uint32_t)A[k + 1] << 8;
        a4 |= (uint32_t)A[k + 2] << 16;
        a4 |= (uint32_t)A[k + 3] << 24;
        __m512i a_bcast = _mm512_set1_epi32((int)a4);          // replicate 4 u8
        __m512i b_vec   = _mm512_loadu_si512((const void*)(B + (size_t)k * 16)); // 64 bytes = 16 lanes * 4 bytes/lane
        acc0 = _mm512_dpbusd_epi32(acc0, a_bcast, b_vec);
    }
    _mm512_storeu_si512((void*)C, acc0);
}
```
2.2D Convolution
```cpp
// Computes Y[16] for a single output pixel, accumulating over (kh,kw,cin).
// Assumes input is u8 HWC: X[(ih*W + iw)*C_in + cin]
// Weights are packed: W_pack[p][16], where p iterates over all (kh,kw,cin) in the same order.
// Ktotal = KH*KW*C_in must be multiple of 4 for this simple kernel.
void conv2d_u8s8s32_pixel_16oc_vnni(
    const uint8_t* X, int H, int W, int C_in,
    int ih0, int iw0,                 // top-left input corner for this output pixel
    const int8_t* W_pack,             // [Ktotal][16] packed
    int32_t* Y16,                     // [16] accumulator/output
    int KH, int KW)
{
    const int Ktotal = KH * KW * C_in;
    __m512i acc = _mm512_loadu_si512((const void*)Y16);

    int p = 0;
    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            const uint8_t* xptr = X + ((size_t)(ih0 + kh) * W + (iw0 + kw)) * C_in;
            // Iterate cin, but we need to feed dpbusd in chunks of 4 bytes of A.
            for (int c = 0; c < C_in; c += 4, p += 4) {
                uint32_t a4;
                a4  = (uint32_t)xptr[c + 0];
                a4 |= (uint32_t)xptr[c + 1] << 8;
                a4 |= (uint32_t)xptr[c + 2] << 16;
                a4 |= (uint32_t)xptr[c + 3] << 24;
                __m512i a_bcast = _mm512_set1_epi32((int)a4);
                // Need 4 successive packed-weight rows (p..p+3), each is 16 bytes.
                // Together they form 64 bytes laid out as 16 lanes * 4 bytes/lane.
                __m512i b_vec = _mm512_loadu_si512((const void*)(W_pack + (size_t)p * 16));
                acc = _mm512_dpbusd_epi32(acc, a_bcast, b_vec);
            }
        }
    }
    _mm512_storeu_si512((void*)Y16, acc);
}
```
3. Inference Fully Connected
Using above GEMM Impl.
```cpp
// Post-op: y_u8 = clamp_u8( round( (acc + bias) * scale ) + zp )
// Simple scalar requant for clarity; vectorize in production.
static inline uint8_t requant_relu_u8(int32_t v, float scale, int32_t zp) {
    // ReLU
    if (v < 0) v = 0;
    // Scale
    float f = (float)v * scale + (float)zp;
    // Round to nearest
    int32_t q = (int32_t)lrintf(f);
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return (uint8_t)q;
}

// Computes 16 outputs for one input vector x[K] against weight matrix W[K][N] packed in 16-wide blocks.
// Produces u8 activations for next layer.
void fc_u8s8_u8_1x16_vnni(
    const uint8_t* x,             // [K]
    const int8_t*  W_pack,        // [K][16] contiguous, K-major
    const int32_t* bias,          // [16]
    uint8_t* y,                   // [16]
    int K,
    float scale,                  // (output_scale / (input_scale*weight_scale)) etc.
    int32_t zp)                   // output zero-point
{
    int32_t acc[16];
    // init acc with bias
    for (int i = 0; i < 16; i++) acc[i] = bias[i];
    gemm_u8s8s32_1x16_vnni(x, W_pack, acc, K);
    for (int i = 0; i < 16; i++) {
        y[i] = requant_relu_u8(acc[i], scale, zp);
    }
}
```
4. Signal Processing correlation
```cpp
// Returns sum_{i=0..K-1} x_u8[i] * h_s8[i], K % 4 == 0
int32_t dot_u8s8_vnni(const uint8_t* x, const int8_t* h, int K)
{
    __m512i acc = _mm512_setzero_si512(); // 16 lanes of s32 partial sums
    // Process 64 taps per iteration: 16 lanes * 4 bytes/tap-group = 64 bytes
    for (int i = 0; i < K; i += 64) {
        __m512i x64 = loadu_64x_u8(x + i);
        __m512i h64 = loadu_64x_i8(h + i);
        acc = _mm512_dpbusd_epi32(acc, x64, h64);
    }
    // Horizontal sum 16 lanes
    __m256i lo = _mm512_castsi512_si256(acc);
    __m256i hi = _mm512_extracti64x4_epi64(acc, 1);
    __m256i sum256 = _mm256_add_epi32(lo, hi);
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));
    // reduce 4 lanes
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}
```
