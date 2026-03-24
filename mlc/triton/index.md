# Learning Triton 
## Structure
- **Week 1 Triton’s model in compiler terms**
- **Week 2 Memory, tiling, and performance semantics**
- **Week 3 Patterns for DL ops & integration**
- **Week 4 Advanced topics, codebase study, and synthesis**


## Week 1 (Days 1–7): Triton’s Programming & Execution Model

####  Day 1 – Triton in the Landscape

**Concepts to learn**

- Triton’s role:
  - Sits between high-level frameworks (PyTorch) and CUDA/HIP.
  - DSL that abstracts:
    - Thread block configuration
    - Shared memory management (abstracted into loads/stores with caching strategies)
- Key design points:
  - Python as a macro/meta layer.
  - Kernels as decorated Python functions: `@triton.jit`.
  - SPMD: each “program” is an independent instance running on a logical tile.

**Connections (compiler mindset)**

- Relate Triton kernels to:
  - **SPMD IR** (like ISPC / SPIR-V) specialized to GPU block-level SPMD.
  - High-level loop nests lowered to SPMD lanes over tiles.
- Think: “Loop tiling + vectorization” encoded as a GPU-friendly SPMD model.

**Explore / Read**

- Triton documentation: **Introduction / Basics / Programming Model**.
- High-level architecture docs or talks (if available from OpenAI / Triton maintainers):
  - Focus on how Triton maps to LLVM and then to PTX/ROCm.


####  Day 2 – Programs, Grids, and Indexing Semantics

**Concepts to learn**

- `tl.program_id(axis=…)`:
  - Defines which logical tile this kernel instance is responsible for.
- Grid definition:
  - 1D vs 2D vs 3D grids at launch.
- Tiling:
  - `BLOCK_SIZE`, `BLOCK_M`, `BLOCK_N`, `BLOCK_K`.
- Indexing with `tl.arange` and pointer arithmetic vs tensor indexing.

**Connections**

- View `tl.program_id` as the outermost tiled loop indices.
- View `tl.arange` as innermost vector lanes within a tile.
- Compare with:
  - Polyhedral loop tiling and then mapping tiles to CTAs + lanes.
  - OpenMP target teams + parallel + simd mapping.

**Explore / Read**

- Triton docs: **Blocked and fused kernels** / indexing examples.
- Study 2–3 example kernels:
  - Simple vector add.
  - 2D add or elementwise op over matrices.
  - Pay attention to: **grid configuration** vs internal indexing.


####  Day 3 – Memory Model & `tl.load`/`tl.store`

**Concepts to learn**

- Global memory access model:
  - `tl.load(ptrs, mask=…, other=…)`
  - `tl.store(ptrs, values, mask=…)`
- Coalescing and alignment:
  - How pointer layouts and `tl.arange` patterns translate to coalescing.
- Masking semantics for boundaries.

**Connections**

- Compare to:
  - Vector predication (masked loads/stores in AVX-512/SVE).
  - GPU memory transactions for contiguous vs strided accesses.
- Reason about:
  - How to design `ptrs` so threads in a warp access consecutive addresses.

**Explore / Read**

- Docs/tutorial sections on:
  - Masks and boundary conditions.
  - Examples that show coalesced vs strided memory patterns.
- Consider drawing small 2D memory diagrams mapping `tl.arange` to addresses.


####  Day 4 – SSA-like Semantics, Control Flow, and Types

**Concepts to learn**

- Triton kernel body semantics:
  - Mostly **functional / SSA-flavored**.
  - Scalar vs tensor-shaped values, broadcasting rules.
- Control flow:
  - Conditionals via masks vs explicit `if` (and how they lower).
- Types and dtypes:
  - `tl.float32`, `tl.float16`, `tl.bfloat16`, `tl.int*`, `tl.bool`.
  - Implicit promotions; which ops support which dtypes.

**Connections**

- Map to:
  - LLVM IR: typed SSA; scalar + vector types.
  - Predicated execution in SIMD/GPUs.
- Think: Triton kernel is like a **typed, SPMD DSL** that will be lowered to an SSA IR with predication.

**Explore / Read**

- Type system section in docs.
- An example kernel involving:
  - Mixed integer and float arithmetic.
  - Branching implemented via masks.


####  Day 5 – Reductions Semantics

**Concepts to learn**

- Reductions in Triton:
  - `tl.sum`, `tl.max`, etc. over a dimension within a program.
  - Partial reduction patterns (within a tile; cross-tile requires more design).
- Numerical stability aspects with reductions:
  - E.g., reduction order, associativity, and FP error.

**Connections**

- Compare to:
  - Vectorized reduction lowering passes in compilers.
  - Warp-shuffle-based vs shared-memory-based reduction patterns in CUDA.
- Consider:
  - Triton’s reductions are **intra-program/intra-tile**; cross-program aggregates require explicit orchestration (multiple kernel calls or special patterns).

**Explore / Read**

- Tutorials for row-wise sum, softmax, or norm computing mean/variance.
- Pay attention to:
  - How reductions iterate over `K` dimension within a tile.
  - How they avoid serial loops where possible.


####  Day 6 – Compilation Pipeline & Backend Architecture

**Concepts to learn**

- Rough pipeline:
  1. Python AST → Triton IR (or high-level representation).
  2. Triton IR → LLVM IR.
  3. LLVM IR → PTX (NVPTX), AMDGCN, etc.
- Optimization stages:
  - CSE, LICM, vectorization-like passes (SPMD aware), memory scheduling.
- Integration with external compilers/drivers:
  - JIT compilation, caching, re-use across runs.

**Connections**

- Place Triton among:
  - MLIR / XLA / TVM / Halide / OpenMP offload.
- Think about:
  - Where you would plug in custom passes (e.g., tiling, fusion, scheduling).
  - How Triton differs from generic “GPU dialect” pipelines (e.g., MLIR’s GPU dialect).

**Explore / Read**

- Any available doc or conference talk on Triton’s internal compiler pipeline.
- If source is available, skim:
  - The IR definition.
  - Pass manager or pipeline registration points.


####  Day 7 – Week 1 Synthesis

**Concepts to consolidate**

- Triton model as:
  - Tiled SPMD DSL over a **block** (program) level.
  - Predicated vectorization inside tiles.
- Understand the mapping:
  - `program_id` ↔ tile loops / blocks
  - `arange` + `load/store` ↔ lanes / coalesced accesses
  - `tl.sum` / `tl.max` ↔ intra-tile reductions

**Self-questions**

- Can you describe Triton to another compiler engineer in 5–10 minutes?
- Do you see how you might lower a high-level loop nest to Triton?

**Suggested reading**

- Re-skim the docs with this mental model; mark open questions related to:
  - Alias analysis, scheduling, or memory optimization opportunities you suspect exist.


## Week 2 (Days 8–14): Memory, Tiling, and Performance Semantics

####  Day 8 – GPU Memory Hierarchy & How Triton Exposes It

**Concepts to learn**

- Memory levels:
  - Global DRAM, L2, SM-local caches, registers.
  - (Shared memory is abstracted; Triton’s loads/stores + `num_stages` hint pipeline).
- Locality strategies:
  - Blocking along K in matmul.
  - Reuse of tiles in registers.

**Connections**

- Analogy to:
  - Classical cache tiling (L1/L2-blocked matrix multiplication).
  - Multi-level tiling passes in polyhedral compilers.
- Think about:
  - Where Triton leaves the decision up to the programmer vs compiler.

**Explore / Read**

- Official matmul/attention tutorials (focusing on memory reuse discussion).
- Any docs/blogs describing `num_stages` and prefetching behavior.


####  Day 9 – Tiling Space: BLOCK_M, BLOCK_N, BLOCK_K, num_warps

**Concepts to learn**

- Tiling design:
  - Trade-offs between larger tiles (more reuse, more registers) and occupancy.
- `num_warps`:
  - Effective SPMD vector width within a program.
  - Interplay with block shapes and register usage.
- Basic occupancy intuition:
  - Relationship to registers/thread, shared mem, and active blocks/SM.

**Connections**

- Compare:
  - Classic register tiling vs L1 tiling vs L2 tiling in GEMM.
  - Unrolling factors in auto-vectorization.
- Consider:
  - This is essentially a **parameterized schedule** (like TVM/Halide) but expressed manually.

**Explore / Read**

- Triton tuning examples (e.g., sweep over tile sizes and `num_warps`).
- Any guidance in docs on heuristic parameter choices for GEMM/attention.


####  Day 10 – Bank Conflicts, Coalescing, and Strides

**Concepts to learn**

- Memory transaction granularity on GPUs.
- Access patterns:
  - Unit-stride vs strided loads.
  - Effect on memory coalescing and bandwidth.
- Bank conflicts (at least conceptually) in shared memory or caches, and how that reflects in pointer arithmetic in Triton.

**Connections**

- Think in terms of:
  - Data layout transformations in classic compilers (AoS vs SoA).
- Understand:
  - In Triton, a lot of “layout decisions” are simply pointer arithmetic expressions you write.

**Explore / Read**

- Examples of “good” vs “bad” pointer layouts in Triton docs or blog posts.
- If available: GPU performance docs (NVIDIA/AMD) to remind yourself of transaction rules.


####  Day 11 – Softmax and Normalization as Case Studies

**Concepts to learn**

- Softmax:
  - Two-pass pattern (max, then exp+sum).
  - Numerical stability.
  - Memory traffic analysis: reading X multiple times vs buffering.
- LayerNorm / RMSNorm:
  - Per-row reductions (mean/var).
  - Multi-step pipeline (compute stats, normalize, apply affine).

**Connections**

- Analyze:
  - How many bytes are transferred vs flops.
  - Identify if the kernel is clearly memory-bound; think about roofline.
- Compare:
  - Doing these ops as separate kernels vs fused kernels.

**Explore / Read**

- Official Triton softmax / layernorm examples.
- PyTorch / xFormers / FlashAttention Triton kernels (if open-sourced) for:
  - Implementation variants and performance justification.


####  Day 12 – GEMM/Matmul Internals

**Concepts to learn**

- Matmul tiling:
  - Blocking along M, N, K.
  - Register tiles, outer loops over K.
- Accumulation strategies:
  - Accumulate in fp32, store in fp16/bf16.
- Data layout variations:
  - A in row-major, B in col-major, etc.
  - Impact of transposed variants.

**Connections**

- Map:
  - High-performance GEMM strategies you know (Goto, BLIS) to Triton expressions.
- Consider:
  - How you’d express a micro-kernel + packing strategy in Triton.

**Explore / Read**

- Triton matmul tutorial.
- Any “high-performance matmul in Triton” blog or talk.


####  Day 13 – Fusion Patterns and Trade-offs

**Concepts to learn**

- Fusion opportunities:
  - Matmul + bias + activation.
  - Attention: QK^T, softmax, AV.
- Downsides of over-fusion:
  - Register pressure, code size, compilation time.
  - Reduced occupancy due to resource usage.

**Connections**

- Compare to:
  - Operator fusion passes in XLA/TVM.
  - Your notions of granularity in loop fusion and stage partitioning.
- Think:
  - Fusion as a schedule parameter, not a binary “always fuse” rule.

**Explore / Read**

- Fused attention kernel write-ups (e.g., FlashAttention, xFormers).
- If Triton implementations are available, inspect:
  - Where they draw the line on what to fuse.


####  Day 14 – Week 2 Synthesis

**Concepts to consolidate**

- Ability to **qualitatively** reason:
  - Is this kernel memory- or compute-bound?
  - Does this tiling make good use of caches/registers?
- Basic sense of:
  - How `BLOCK_*`, `num_warps`, `num_stages` influence performance.

**Self-questions**

- If given a new op (e.g., some batched reduction + elementwise), can you:
  - Sketch a tiling strategy?
  - Argue about memory traffic and potential fusion?

**Suggested reading**

- Roofline model refresher (for GPUs).
- Any documentation describing recommended tuning approaches for Triton.


## Week 3 (Days 15–21): Deep Learning Patterns & Integration

####  Day 15 – Triton + PyTorch Tensor Semantics

**Concepts to learn**

- Relationship between:
  - PyTorch `Tensor` (shape, strides, dtype, device)
  - Triton kernel arguments (raw pointers + shape/stride ints).
- Handling:
  - Non-contiguous tensors (e.g., transposed).
  - Stride-based indexing vs assuming contiguity.

**Connections**

- Map:
  - N-dimensional tensor indexing `i0, i1, …` to pointer offsets using strides.
- Consider:
  - How you’d design a kernel that preserves layout agnosticism vs one that assumes contiguity for performance.

**Explore / Read**

- Triton docs/test code on integration with PyTorch.
- Example kernels that accept strides and handle generic layouts.


####  Day 16 – Autograd & Custom Ops (Integration Semantics)

**Concepts to learn**

- PyTorch custom ops:
  - `torch.autograd.Function`.
  - `forward`/`backward` semantics.
- Strategy:
  - Forward in Triton, backward in PyTorch vs both in Triton.
- Gradient stability and FP16/BF16 concerns.

**Connections**

- Think:
  - This is integration at the IR boundary of two systems: Triton-produced kernels and PyTorch’s autograd graph.
- Consider:
  - Designing kernels so that necessary auxiliary information for backward is cheap to store/compute.

**Explore / Read**

- PyTorch custom extension / custom autograd docs.
- Any examples in Triton docs of autograd usage.


####  Day 17 – Canonical DL Ops: Attention

**Concepts to learn**

- Attention decomposition:
  - QK^T / √d: batched matmul with scaling.
  - Softmax (row-wise).
  - AV: another batched matmul.
  - Optional: dropout, causal masking.
- Complexity and memory:
  - O(B · H · L² · D) patterns; quadratic in sequence length.

**Connections**

- Analyze:
  - Why fused attention kernels (FlashAttention-style) reduce memory traffic.
  - How blocking over sequence length L (and maybe heads) trades storage for re-computation.

**Explore / Read**

- FlashAttention papers/blog posts (focus on algorithmic strategy).
- Any available Triton implementations of attention kernels.
- Relate:
  - Their blocking scheme to your Week 2 tiling knowledge.


####  Day 18 – Canonical DL Ops: Normalization Family

**Concepts to learn**

- LayerNorm, GroupNorm, RMSNorm:
  - Different reduction domains.
  - Parameterization (gamma, beta).
- Implementation patterns:
  - Row-wise vs group-wise reductions.
  - One-pass vs two-pass (stats then normalize).

**Connections**

- See them as:
  - Structured reductions with simple elementwise post-processing.
- Compare:
  - Implementation-level differences vs algorithmic ones (they’re structurally similar but with different grouping).

**Explore / Read**

- Any Triton normalization kernels (from frameworks/libraries).
- Think about:
  - Where to stash intermediate statistics, and how to minimize passes.


####  Day 19 – Mixed Precision & Numerical Considerations

**Concepts to learn**

- Dtype rules in Triton:
  - Where accumulation is done in fp32 vs fp16/bf16.
- Trade-offs:
  - Accuracy vs performance vs memory.
- Overflow/underflow scenarios:
  - In softmax, matmul with large norms, etc.

**Connections**

- Compare:
  - Your mental model of rounding error and ULP analysis with what’s necessary in DL practice.
- Think:
  - Where you’d enforce upcasting (input fp16 → accumulation fp32).

**Explore / Read**

- Mixed precision training background (NVIDIA whitepapers, etc.).
- Any Triton examples that emphasize dtype handling.


####  Day 20 – Kernel Parameterization & (Semi-)Automatic Tuning

**Concepts to learn**

- Parameterizing kernels:
  - Expose `BLOCK_*`, `num_warps`, etc. as compile-time constants.
- Simple autotuning loops:
  - Exhaustive search over small parameter sets.
  - Heuristics per architecture and problem sizes.
- Caching compiled variants.

**Connections**

- Compare:
  - Auto-scheduling and auto-tuning in TVM / Halide / MLIR-based systems.
- Consider:
  - What a cost model for Triton might look like.

**Explore / Read**

- Triton docs or examples mentioning autotuning.
- Any external libraries that use Triton with param sweeps (search in open-source repos).


####  Day 21 – Week 3 Synthesis

**Concepts to consolidate**

- Patterns for:
  - Attention, matmul, normalization, softmax.
- Integration understanding:
  - Where Triton fits in a PyTorch model’s hot path.

**Self-questions**

- If given a new compound op (e.g., some fused conv+norm+activation), can you:
  - Identify candidate fusion boundaries?
  - Sketch a kernel schedule?

**Suggested reading**

- Any codebases you plan to deeply inspect in Week 4 (e.g., xFormers, FlashAttention, other Triton-heavy repos).


## Week 4 (Days 22–30): Advanced Topics, Codebase Study, and Synthesis

####  Day 22 – Advanced Tiling & Multi-Dimensional Grids

**Concepts to learn**

- Multi-dimensional `program_id`s:
  - Joint tiling across batch, head, and spatial/sequence dims.
- Hierarchical tiling:
  - Multi-axis blocking to balance reuse and occupancy.
- Handling large high-dimensional tensors:
  - `[B, H, L, D]`, `[N, C, H, W]`, etc.

**Connections**

- Compare:
  - Multi-dimensional loop tiling in polyhedral models.
- Consider:
  - How you might systematically search over multi-axis tilings (meta-scheduling ideas).

**Explore / Read**

- Attention kernels or other high-D kernels using 2D/3D grids.
- Study their grid parameterization and indexing formulas.


####  Day 23 – Inspecting Triton-Generated IR / PTX

**Concepts to learn**

- How to:
  - Dump or view intermediate IR / final PTX/SASS for a Triton kernel.
- Key things to inspect:
  - Register count, instruction mix, memory ops, divergence.
- Relate high-level kernel changes to low-level changes:
  - E.g., unrolling, vectorization, predication patterns.

**Connections**

- Map:
  - Triton IR constructs to LLVM IR constructs.
  - LLVM IR to PTX instructions (loads/stores, FMAs, etc.).
- Think:
  - Where you might want more control vs compiler heuristics.

**Explore / Read**

- Any developer documentation for inspecting Triton’s generated code.
- Use simple kernels and compare:
  - Slight changes in indexing or tiling → differences in PTX.


####  Day 24 – Real-World Kernel Case Study I

**Concepts to learn**

- Take one substantial kernel from open source (e.g., a Triton attention kernel):
  - Understand its structure in depth.
- Key aspects to examine:
  - Tiling along each dimension.
  - Use of `num_warps`, `num_stages`.
  - Handling of masks (causal, padding).
  - Dtype manipulations and accumulations.

**Connections**

- Critically evaluate:
  - Where they might be leaving performance on the table.
  - What assumptions they encode (sequence length ranges, head dims, etc.).

**Explore / Read**

- Source of a known Triton kernel implementation:
  - Fused attention, high-performance matmul, multi-head attention, etc.
- Annotate for yourself:
  - “Schedule” description in plain English.


####  Day 25 – Real-World Kernel Case Study II

**Concepts to learn**

- Pick a second domain:
  - Normalization family.
  - Fused MLP block (linear → activation → linear).
  - Some non-trivial reduction + elementwise mixture.
- Analyze similarly:
  - Indexing design.
  - Tiling vs. fusion choices.
  - Handling of corner cases & broadcasting.

**Connections**

- Compare with Case Study I:
  - Different ops, but many shared patterns (tiling, dtype handling).
- Think:
  - Could these be described by a common meta-schedule?

**Explore / Read**

- Another open-source Triton kernel codebase.
- Document your understanding in a short note (even just for yourself).


####  Day 26 – Cross-Architecture Considerations

**Concepts to learn**

- How Triton targets:
  - Different GPU architectures (NVIDIA vs AMD).
- Architecture-specific considerations:
  - Warp size differences.
  - Cache and memory behavior differences.
- Portability vs tuning:
  - How far can one schedule go across architectures?

**Connections**

- Compare:
  - Multi-target compiler design (LLVM’s backends).
- Consider:
  - Which optimizations are backend-agnostic vs architecture-specific.

**Explore / Read**

- Any Triton documentation or discussions about AMD support, architecture-conditional tuning.
- General GPU hardware docs to refresh yourself on differences.


####  Day 27 – “If I Were Designing Passes for Triton…”

**Concepts to learn**

- Identify potential compiler improvements:
  - Auto-tiling, auto-fusion, better cost modeling, improved register allocation hints.
- Where manual scheduling could be replaced by:
  - Auto-scheduling / meta-scheduling.

**Connections**

- Use your compiler background to:
  - Propose (for yourself) how you’d extend Triton’s IR or passes.
- Example thoughts:
  - A high-level schedule DSL layered on Triton.
  - Analyses to infer safe fusion or reordering.

**Explore / Read**

- MLIR/TVM auto-scheduling papers for inspiration.
- Reflect on which ideas port well to Triton’s domain-specific nature.


####  Day 28 – Synthesis of Patterns into a Mental “Design Cookbook”

**Concepts to learn**

- Distill:
  - Common kernel design patterns:
    - Tiled matmul-like.
    - Tiled reduction + broadcast.
    - Elementwise + gather/scatter.
- Map:
  - Each pattern to typical DL use-cases.

**Connections**

- Construct for yourself:
  - A mapping: “Given op X with shape property Y, likely design pattern is Z, with these tiling and dtype considerations.”
- Treat this as a high-level **design cookbook**.

**Explore / Read**

- Look back at all sampled kernels and your notes.
- Abstract the common structure.


####  Day 29 – Formalize Knowledge: Write a Short Internal Doc

**Concepts to consolidate**

- Triton summary aimed at compiler engineers:
  - Model, pipeline, performance view.
  - Design idioms and anti-patterns.

**Task**

- Write an internal-style doc (even if only for yourself) covering:
  - Overview of Triton programming model and compilation pipeline.
  - Kernel design patterns for common DL ops.
  - Performance-tuning heuristics.
  - Open research/engineering questions you see (e.g., auto-scheduling).

**Connections**

- This will be your own “mini-spec”/guide, capturing:
  - **What to learn** you’ve gone through over 30 days, but distilled and systematized.


####  Day 30 – Reflection and Next Steps

**Concepts to reflect on**

- Where you have deep understanding vs surface-level familiarity.
- Which pieces intersect most with your compiler interests:
  - IR, passes, scheduling, cost modeling, backends, etc.

**Self-questions**

- If you had to:
  - Design a new Triton kernel for a novel op tomorrow, can you reason through:
    - Tiling, memory, dtypes, and integration?
- If you had to:
  - Contribute to Triton’s compiler pipeline, where would you start?

**Next steps (directional)**

- Consider:
  - Contributing small improvements/experiments to Triton.
  - Prototyping auto-scheduling heuristics on top of Triton.
  - Exploring integration of Triton with other IR stacks (e.g., MLIR frontends).


## Exploration List
- Triton official docs and tutorials (read with a **compiler’s eye**).
- OpenAI / community blog posts on Triton and performance case studies.
- FlashAttention papers / blogs (for fused attention design).
- TVM / Halide / MLIR auto-scheduling literature for context.
- GPU architecture whitepapers (NVIDIA/AMD) focusing on:
  - Memory hierarchy
  - Warp scheduling
  - Tensor cores / MMA units
