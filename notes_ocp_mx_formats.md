
### https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf


### What “FP8 format” means 
**FP8** is typically used as a *storage + compute* format for matrix math, always paired with **explicit scaling metadata** so tensors keep usable dynamic range.

core Idea of “mixed precision”:
- Inputs stored/processed in **FP8** for speed/bandwidth
- Accumulation performed in **FP16** or **FP32** for numerical stability
- Outputs often returned as **BF16/FP16** (or re-quantized to FP8 with new scales)

+ Common FP8 formats
- **E4M3** (4 exponent bits, 3 mantissa bits):  
  More precision, less range => often preferred for activations/weights where values are not extremely large.
- **E5M2** (5 exponent bits, 2 mantissa bits):  
  More range, less precision => often preferred for values with larger magnitude variation (some gradients / intermediates).

+ scaling is applied because:
- distributions vary per layer and per batch
- outliers can dominate the max range
- small values can disappear without careful scaling

+ Scaling granularity (“how many values share one scale?”)
- **Per-tensor scale:** one scale for the entire tensor  
  - Pros: minimal overhead  
  - Cons: poor handling of channel-wise or block-wise variation
- **Per-channel scale:** one scale per output channel (common for weights)
- **Per-block / per-tile scale:** one scale per fixed-size group (common for high-performance FP8 GEMM)  
  Examples of block sizes: 16, 32, 64, 128 elements (varies by kernel/hardware)
  + smaller blocks → better accuracy, more scale overhead.

+ Example: block-scaled FP8 quantization 
Assume a block of 8 FP16 values (activations), and we choose **one scale per block**.
1. Original (FP16) `x = [0.50, -1.20, 0.10, 0.00, 2.00, -0.30, 0.70, -0.80]`
2. scale e.g. **max-based scaling**:
- `amax = max(|x|) = 2.00`
- maximum magnitude representable by the chosen FP8 format is `fp8_max`.
- Then: `s = amax / fp8_max`

3. normalize and encode
For each element:
- `u[i] = x[i] / s`
- `q[i] = fp8_encode(u[i])`

4. store payload + metadata
Store:
- FP8 block: `q[0..7]`
- scale: `s` (FP16/FP32)

5. reconstruct at compute time
- `x_hat[i] = s * fp8_decode(q[i])`

Then GEMM uses `x_hat` during FP8 tensor-core operations with higher-precision accumulation.

- rounding modes, stochastic rounding, saturation behavior, separate scales per row/column tile to match GEMM tiling.


+ Example: FP8 GEMM with FP16 accumulation 
For `Y = A * B`:
- `A`: FP8 payload + scales (block-scaled)
- `B`: FP8 payload + scales (block-scaled)
- Multiply: FP8 × FP8
- Accumulate: FP16 (or FP32)
- Output: BF16/FP16 (optionally FP8 with new scales)

Pseudocode sketch:

1. Load `(A_q, A_s)` and `(B_q, B_s)`
2. On-the-fly dequantization is fused into the kernel math:
   - use `A_s` and `B_s` to scale products appropriately
3. Accumulate partial sums in FP16/FP32
4. Write `Y` in BF16/FP16, or re-quantize to FP8

This is “mixed precision” because:
- **storage/compute** is low precision (FP8)
- **accumulation/output** is higher precision


+ trade-offs
- **Accuracy vs overhead:** more scales (smaller blocks) means more metadata and memory traffic.
- **Choice of E4M3 vs E5M2:** precision vs range trade-off, often layer-dependent.
- **Scale selection policy:** max-based is simple but sensitive to outliers; percentile/running-statistics methods can be more robust.
- **Interoperability:** a standard memory layout for payload+scales enables cross-vendor kernels to consume the same tensor format.


+ What to do for 
- Keep accumulation at FP16/FP32 for stability.
- Use per-block scaling for activations to reduce quantization error.
- Use per-channel/per-block scaling for weights depending on kernel support.
- Validate with layer-wise error metrics (cosine similarity, MSE) and end-to-end task accuracy.
