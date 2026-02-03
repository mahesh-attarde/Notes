
### https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

## FP MX
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

## INT MX

+ What “scaled INT8/INT4 format” means
Scaled integer quantization represents real-valued tensors using:
- **Integer payload** (`q[i]` in INT8 or INT4)
- **Scale(s)** (`s`), typically FP16/FP32
- (Optionally) **Zero-point(s)** (`z`), especially for asymmetric quantization

+ Reconstruction 
	* Symmetric quantization (common for weights)
		- `x[i] ≈ s * q[i]`
		- where `q[i] ∈ [-127,127]` for INT8 (or `[-7,7]` for signed INT4)

    * Asymmetric quantization (common for activations)
		- `x[i] ≈ s * (q[i] - z)`
		- where `z` shifts the representable range to better match non-zero-centered distributions

+ integer formats are usually paired with:
	- **INT32 accumulation** for dot products
	- Rescaling back to **BF16/FP16** or **INT8** depending on pipeline

+ Scaling granularity
- **Per-tensor:** one scale for entire tensor  
- **Per-channel (per output channel):** very common for weight matrices
- **Per-group / per-block:** common for INT4 weight-only compression (group size 32/64/128, etc.)

+ Smaller groups better preserve accuracy but increase scale storage and compute overhead.
+ example: INT8 per-channel weight quantization
For a linear layer weight matrix `W[out, in]`, quantize per output channel is 
* compute per-channel scale
	For each output channel `c`:
	- `amax_c = max_i |W[c,i]|`
	- `s_c = amax_c / 127`

* quantize to INT8
	For each element:
	- `Q[c,i] = round(W[c,i] / s_c)`
	- clip to `[-127,127]`

* reconstruct
- `W_hat[c,i] = s_c * Q[c,i]`

* Eg.
Suppose one channel has 4 weights:
- `W = [0.50, -1.00, 0.25, -0.75]`
- `amax = 1.00`
- `s = 1.00 / 127 ≈ 0.007874`
Quantize:
- `Q ≈ [ round(0.50/0.007874)=64,
         round(-1.00/0.007874)=-127,
         round(0.25/0.007874)=32,
         round(-0.75/0.007874)=-95 ]`

Reconstruction:
- `W_hat ≈ [0.504, -1.000, 0.252, -0.748]`


+ example: INT4 group-wise quantization (common for weight-only inference)

	* INT4 has fewer levels than INT8, so it often uses **group-wise scales** (a compromise between per-tensor and per-channel).
		Let group size be 32 elements within a row.
		For group `g`:
		- `amax_g = max(|W_g|)`
        - `s_g = amax_g / 7`  (for signed INT4 in [-7,7])
        - `Q_g = round(W_g / s_g)` clipped to [-7,7]

        Store:
        - packed INT4 nibbles (2 values per byte)
        - one scale per group (FP16/FP32)

+ BIG PRO: reduces memory bandwidth for large models.


+ Integer GEMM: mixed precision compute pipeline

### INT8 activation × INT8 weight 
For `y = xW^T`:
- `x_q`: INT8 (often asymmetric, with scale `s_x` and zero-point `z_x`)
- `W_q`: INT8 (often symmetric, per-channel scale `s_w[c]`)
- Dot product accumulates into **INT32**:
  - `acc[c] = Σ (x_q[i] - z_x) * W_q[c,i]`  (if using asymmetric activations)
- Rescale:
  - `y_fp16[c] = (s_x * s_w[c]) * acc[c]`  (plus bias, etc.)

### INT4 weight-only inference 
- Activations may be BF16/FP16/INT8
- Weights INT4 group-wise
- Kernel dequantizes weights on-the-fly and accumulates in FP16/FP32 or INT32 depending on design


+ trade-off
- **INT8**: good balance, widely supported, strong accuracy for many models.
- **INT4**: major compression/bandwidth gains, but more sensitive to outliers; requires careful scaling and often per-group metadata.
- **Asymmetric quantization**: better for non-zero-centered activations, but adds zero-point handling.
- **Per-channel/group scaling**: improves accuracy but increases metadata and kernel complexity.

+ what to do when
- For weights: prefer **symmetric** + **per-channel** (INT8) or **per-group** (INT4).
- For activations: consider **asymmetric** INT8 if distributions are not centered at 0.
- Accumulate in INT32 (integer path) or FP16/FP32 (hybrid path).
- Validate layer-wise error + end-to-end task metrics; watch out for outlier channels/groups.
