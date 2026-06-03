# FLIP Fluid CUDA Kernel Executive Summary

This document summarizes the CUDA kernels defined in `flip_fluid_cuda.cu`. It focuses on the role of each kernel in the FLIP/PIC solver pipeline, and notes where a specific implementation choice is meaningful.

## Overview

The CUDA implementation follows the standard FLIP fluid solver stages:
- integrate particles
- collide geometry and boundary
- transfer particle data to the grid (P2G)
- compute density and pressure
- transfer grid velocity back to particles (G2P)
- optionally push particles apart to avoid overlap
- color particles and cells for visualization
- carve solid obstacles into the grid

Many kernels are simple and directly mirror the CPU logic. Some choose explicit methods for parallel correctness.

## Kernel Summaries

### `integrateKernel`
- Moves particles with their current velocity and applies gravity.
- Simple per-particle Euler integration.

### `collisionsKernel`
- Handles boundary collisions and a circular obstacle.
- Clamps particles back inside the simulation domain and zeroes velocity at walls.
- If a particle is inside the obstacle, its velocity is replaced by the obstacle velocity.

### `savePrevKernel`
- Clears grid velocity accumulators (`u`, `v`) and saves the previous grid velocities into `prevU` and `prevV`.
- Also zeroes gradient accumulators `du`, `dv`.
- This prepares the grid for a fresh P2G transfer while retaining the previous state needed for FLIP correction.

### `classifyGridKernel`
- Marks grid cells as solid or air based on the solid mask `s`.
- Simple direct classification.

### `classifyParticlesKernel`
- Converts cells touched by particles from air to fluid.
- This is idempotent: multiple particles may touch the same cell and all writes are the same, so no atomics are needed.

### `p2gKernel`
- Transfers a single velocity component from particles to the MAC grid.
- Uses bilinear weighting from particle position to the four nearby grid faces.
- Writes both weighted velocity contributions and weights into fields `fld` and `fldD` using `atomicAdd`.
- This is the standard scatter-based P2G approach for grid-based PIC/FLIP.

### `normalizeKernel`
- Divides accumulated grid velocity by the corresponding accumulated weight (`fldD`).
- Simple normalization to complete weighted averaging.

### `restoreSolidKernel`
- Restores solid boundary face velocities from the saved previous grid state for cells adjacent to solid boundaries.
- Chosen because the fluid solver uses a MAC grid and enforcing solid-face values directly keeps boundary conditions consistent.

### `densityKernel`
- Computes per-cell particle density using bilinear interpolation of particle positions into the cell-centered density field.
- Uses `atomicAdd` for contributions from multiple particles.

### `restDensityKernel`
- Computes a cell-average rest density by summing densities in fluid cells and counting them.
- Uses atomic addition into two scalar values on the device, then reads them back to the host.

### `pressureInitKernel`
- Initializes pressure to zero and snapshots post-P2G grid velocities into `prevU` and `prevV`.
- The snapshot is needed for later FLIP velocity correction.

### `pressureRBKernel`
- Performs one color pass of red-black Gauss-Seidel pressure relaxation on the MAC grid.
- Only processes interior fluid cells of the selected parity (`color` = red or black).
- Uses in-place updates without atomics because red-black ordering ensures each face has at most one writer per pass.
- This implementation choice is important: it gives deterministic parallel updates and mirrors the CPU lexicographic sweep with two-color ordering.
- It also optionally includes drift compensation based on density deviation.

### `g2pKernel`
- Transfers a single velocity component from the grid back to particles.
- Computes a PIC velocity from current grid field and a FLIP correction from the difference between current and previous velocities.
- Blends PIC and FLIP velocities according to `flipRatio`.
- Only uses valid neighbor faces when the adjacent cell is not pure air, to avoid sampling invalid grid data.

### `particleColorsKernel`
- Updates particle colors for visualization.
- Applies a slight fade effect and sets low-density particles to a blueish color.
- This is a simple display-only kernel.

### `setSciColorDev`
- Converts a scalar value into a perceptual color ramp for cell visualization.
- Used by `cellColorsKernel`.

### `cellColorsKernel`
- Colors grid cells for visualization based on cell type and density.
- Solid cells are gray, fluid cells are mapped through a scientific color ramp, and air cells are black.

### `zeroIntKernel`
- Zeros an integer array.
- Used to prepare particle bin counts before building a spatial hash.

### `countKernel`
- Counts particles per cell in a spatial hash grid.
- Uses atomic increments because many particles may map to the same bin.

### `setLastKernel`
- Computes the end offset of the last particle cell in the prefix sum structure.
- Fills the sentinel entry for the cell index array.

### `scatterKernel`
- Builds a compact particle index list per cell by writing particle IDs into the correct bins.
- Uses atomic increments on the cell cursor.

### `separateKernel`
- Performs one Jacobi-style particle separation pass.
- Each particle reads neighbor positions from a fixed snapshot and writes only its own corrected position and color.
- This is a deliberate parallel alternative to CPU sequential pairwise separation; it makes the update deterministic and avoids races by leaving neighbors to correct themselves in later iterations.
- Color diffusion is computed concurrently based on neighbor colors.

### `carveKernel`
- Carves a circular obstacle into the solid mask and updates grid face velocities inside the obstacle.
- This is the same strategy as CPU obstacle carving: mark interior cells solid and initialize velocities to obstacle velocity.

## Notes on Implementation Choices

- `pressureRBKernel` uses a red-black Gauss-Seidel scheme instead of a simple Jacobi or conjugate gradient solver. That choice improves convergence while remaining race-free in a parallel CUDA context.
- `separateKernel` implements particle separation with a Jacobi-style snapshot update instead of the CPU's in-place sequential push-apart. This avoids data races and preserves correctness in a threaded GPU environment.
- The P2G/G2P transfer kernels use standard bilinear weighting plus atomic accumulation for scatter-based grid writes, which is a conventional GPU-friendly approach.

## Pipeline Behavior

The main simulation loop calls kernels in this order:
1. `integrateKernel`
2. optional particle separation (`pushApart`)
3. `collisionsKernel`
4. `transferToGrid` (`savePrevKernel`, `classifyGridKernel`, `classifyParticlesKernel`, `p2gKernel`, `normalizeKernel`, `restoreSolidKernel`)
5. `updateDensity` (`densityKernel`)
6. `solvePressure` (`pressureInitKernel`, repeated `pressureRBKernel` passes)
7. `transferToParticles` (`g2pKernel`)
8. `updateColors` (`particleColorsKernel`, `cellColorsKernel`)

This ordering matches a standard FLIP solver with GPU-specific parallelization choices.

## Pipeline Flow Chart

```d2
integrate: "Integrate particles\n(`integrateKernel`)" -> separate: "Optional particle separation\n(`pushApart` / `separateKernel`)" -> collisions: "Boundary + obstacle collisions\n(`collisionsKernel`)" -> p2g: "Particle-to-grid transfer\n(`savePrevKernel`, `classifyGridKernel`, `classifyParticlesKernel`, `p2gKernel`, `normalizeKernel`, `restoreSolidKernel`)" -> density: "Density update\n(`densityKernel`)" -> pressure: "Pressure solve\n(`pressureInitKernel`, repeated `pressureRBKernel`)" -> g2p: "Grid-to-particle transfer\n(`g2pKernel`)" -> colors: "Color update\n(`particleColorsKernel`, `cellColorsKernel`)
```
