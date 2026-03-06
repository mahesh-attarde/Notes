## 1. Stencils
### 1. What is a “stencil” computation?

 **stencil** is a computation where:

- You have data laid out on a **regular grid** (1D, 2D, 3D, …).
- To compute the value at one grid point, you read values from a **fixed pattern of neighboring points**.
- This pattern (the “stencil”) is the same for almost all grid points and for all time steps.

Formally, think of an array `A` indexed by space and (optionally) time:

- 1D: `A[i]`
- 2D: `A[x, y]`
- 3D: `A[x, y, z]`
- Time + space: `A[t, x, y, z]`

A stencil update might look like:

```c name=1d_stencil.c
// 1D 3-point stencil
for (int i = 1; i < N-1; i++) {
    B[i] = a * A[i-1] + b * A[i] + c * A[i+1];
}
```

Here, the **stencil pattern** is `{i-1, i, i+1}` around each `i`. The same pattern is applied at each iteration.

Key properties:

- **Regular access pattern**: indices are affine functions of loop variables (like `i-1`, `i`, `i+1`).
- **Locality**: only nearby elements are accessed for each update.
- **Uniformity**: the pattern does not depend on data values, only on indices.

This regularity is what makes stencils very attractive to compilers: they can be analyzed and transformed systematically.

### 2. Simple examples of stencils

#### 2.1 1D stencil (finite difference)

```c name=1d_heat_equation.c
// Discrete 1D heat equation time step
for (int i = 1; i < N-1; i++) {
    u_new[i] = u[i] + alpha * (u[i-1] - 2*u[i] + u[i+1]);
}
```

- Grid: 1D line of points.
- Stencil: accesses neighbors `(i-1, i, i+1)` around each point.
- Application: solving a partial differential equation (PDE) for heat diffusion.

#### 2.2 2D stencil (image blur)

```c name=2d_blur.c
for (int y = 1; y < H-1; y++) {
    for (int x = 1; x < W-1; x++) {
        out[y][x] = (
            in[y-1][x-1] + in[y-1][x] + in[y-1][x+1] +
            in[y  ][x-1] + in[y  ][x] + in[y  ][x+1] +
            in[y+1][x-1] + in[y+1][x] + in[y+1][x+1]
        ) / 9.0f;
    }
}
```

- Grid: 2D image.
- Stencil: 3×3 neighborhood (radius 1 in both x and y).
- Application: image processing / smoothing.

#### 2.3 3D stencil (3D Laplacian)

```c name=3d_laplacian.c
for (int z = 1; z < Z-1; z++) {
    for (int y = 1; y < Y-1; y++) {
        for (int x = 1; x < X-1; x++) {
            out[z][y][x] =
                  in[z][y][x-1] + in[z][y][x+1]
                + in[z][y-1][x] + in[z][y+1][x]
                + in[z-1][y][x] + in[z+1][y][x]
                - 6 * in[z][y][x];
        }
    }
}
```

- Grid: 3D volume.
- Stencil: 6 neighbors in the ±x, ±y, ±z directions (a “7-point stencil” including the center).
- Application: 3D PDEs (e.g., in physics simulations).

---

### 3. Why are stencils important?

Stencils appear in many important domains:

1. **Numerical simulation and scientific computing**
   - Finite difference / finite volume schemes for PDEs:
     - Heat equation, wave equation, Poisson equation, Navier–Stokes, etc.
   - Used in climate modeling, fluid dynamics, astrophysics, seismic imaging.

2. **Image processing and computer vision**
   - Convolution filters (blur, sharpen, edge detection).
   - Many CNN operations (conceptually) are stencil-like, though implementations are more general convolutions.

3. **Signal processing**
   - 1D filters, smoothing, differentiation operators.

Because these applications are:

- **Compute-intensive**
- **Memory-bandwidth-intensive**
- And often run on **large grids** for **many time steps**

they are a major target for performance optimization in compilers and libraries.

### 4. Core characteristics

From the perspective of a compiler or optimizer, stencil computations have several defining characteristics:

#### 4.1 Regular iteration space

- The loops iterating over the grid are typically **perfectly nested** and affine:
  - Example: `for (i = 1; i < N-1; i++)` or `for (x = 1; x < W-1; x++)`.
- The index space is often a **rectangular** or **box-shaped** domain:
  - 1D: `[1, N-2]`
  - 2D: `[1, W-2] × [1, H-2]`
  - 3D: `[1, X-2] × [1, Y-2] × [1, Z-2]`

This regularity allows the compiler to reason about dependences and data reuse using static analysis (e.g., polyhedral model).

#### 4.2 Local, structured data dependences

- Each output element depends on **a fixed set of nearby inputs**.
- The offsets (like `i-1`, `i+1`, etc.) are **constant** and **small**.
- Dependence distances are known statically, which is crucial for:
  - Loop transformations (tiling, skewing).
  - Parallelization (deciding which iterations can run concurrently).
  - Vectorization (grouping operations on consecutive elements).

#### 4.3 High arithmetic intensity is limited by memory bandwidth

- Many simple stencils perform few arithmetic operations per data element loaded from memory.
- Performance is often **memory-bound**: limited by how fast data can be moved, not by compute.
- Compilers therefore focus heavily on **data locality and reuse**:
  - Cache blocking / tiling.
  - Reuse of values across iterations.
  - Use of on-chip memories (e.g., GPU shared memory).

#### 4.4 Repeated application over time (time-stepping)

Many stencil-based simulations iterate in time:

```c name=stencil_time_step.c
for (int t = 0; t < T; t++) {
    for (int i = 1; i < N-1; i++) {
        u_new[i] = f(u[i-1], u[i], u[i+1]);
    }
    swap(u, u_new);
}
```

- The same stencil pattern is applied at each time step.
- There are **temporal dependences** between time steps, which can also be exploited:
  - Time tiling, wavefront parallelization, etc.


### 5. Stencil “shape” and neighborhood

The **stencil shape** describes which neighboring points are used:

- **Radius**: maximum distance from the center point (e.g., radius 1 in each dimension).
- **Shape types**:
  - **Star-shaped**: neighbors only in axis-aligned directions (e.g., up/down/left/right in 2D).
  - **Box-shaped (full)**: all points in a surrounding box (e.g., 3×3, 5×5 window).
  - **Cross or diamond**: patterns defined by Manhattan distance, etc.

Example (2D, radius 1, star-shaped):

- Uses `(x, y)`, `(x±1, y)`, `(x, y±1)`.

Example (2D, radius 1, box-shaped / 3×3):

- Uses `(x+i, y+j)` for `i, j ∈ {-1, 0, 1}`.

For compilers, these shapes influence:

- How much data must be loaded for each tile.
- How large the **halo / ghost regions** must be at tile or process boundaries.
- How to schedule computations to maximize reuse.


### 6. Boundary conditions

Stencil formulas are not directly valid at the boundary of the domain, e.g.:

- For `i = 0` or `i = N-1`, `i-1` or `i+1` are out of bounds.

Common strategies:

1. **Skip boundaries** and handle them separately:
   - Have loops from `1` to `N-2` and special code for edges.

2. **Dirichlet boundary conditions**:
   - Fix boundary values to some constant (e.g., 0 or a known function).

3. **Neumann boundary conditions**:
   - Set derivatives at boundaries to some value, implemented through mirrored accesses.

4. **Periodic boundary conditions**:
   - Wrap around: `i = -1` maps to `N-1`, etc.

Why this matters to compilers:

- Boundary handling often breaks the simple regular loop structure.
- Compilers or DSLs may:
  - Generate separate loops for “interior” (fast path) and “boundary” (slow path).
  - Extend arrays with halo cells so that the same stencil code can run everywhere.


### 7. Why compilers care specifically about stencils

Stencils are such a common and well-structured pattern that compilers and DSLs often include **special support**:

1. **Recognition / detection**:
   - Analyze loops and array accesses to see if they match a stencil pattern (affine subscripts with small constant offsets).
   - Once recognized, the compiler can use specialized transformations.

2. **Stencil-aware optimizations**:
   - **Spatial tiling**:
     - Partition the grid into blocks to keep data in cache or shared memory.
   - **Temporal tiling**:
     - Compute multiple time steps for a region before moving on, to reuse data while it’s still in fast memory.
   - **Vectorization and SIMT mapping**:
     - Map iterations to SIMD lanes or GPU threads efficiently.
   - **Communication optimization**:
     - In distributed-memory settings, organize halo exchanges and overlap communication with computation.

3. **Domain-Specific Languages (DSLs) and IRs**:
   - Stencil computations are often expressed in high-level DSLs, and compiled through:
     - Stencil-specific intermediate representations (IR).
     - Specialized back ends for CPUs, GPUs, and clusters.

Examples :

```c name=stencil_dsl_example.dsl
// High-level DSL-like notation
u.new[x, y] = 0.25 * (
    u[x, y] +
    u[x-1, y] + u[x+1, y] +
    u[x, y-1] + u[x, y+1]
);
```

The compiler can interpret this as:

- Iteration domain: interior of 2D grid.
- Stencil radius: 1.
- Shape: 5-point star.
- Dependences: from current time step values of neighbors.

Then it can choose how to schedule/optimize it for a given hardware target.

