# FLIP Fluid Simulator — CPU (C++) and CUDA ports

A 2-D FLIP (Fluid-Implicit-Particle) incompressible fluid simulator. The CPU
version (`flip`) runs the whole pipeline single-threaded; the CUDA version
(`flip_cuda`) ports every pipeline stage to GPU kernels. Both share the exact
same grid/particle layout and an identical per-stage timing + benchmark harness,
so their performance can be compared fairly.

> **Platform:** the CUDA build requires a Linux machine with an NVIDIA GPU and
> the CUDA toolkit. The CPU build needs X11 + legacy OpenGL (`-lGL -lGLX -lX11`),
> i.e. it is also Linux-targeted. Neither builds on macOS.

## Pipeline stages (both builds)

| Code | Stage | CPU function |
|------|-------|--------------|
| T1 | integrate particles | `integrateParticles` |
| T2 | push particles apart (spatial hash) | `pushParticlesApart` |
| T3 | particle collisions | `handleParticleCollisions` |
| T4 | transfer P2G | `transferVelocities(toGrid=true)` |
| T5 | particle density (+ rest density) | `updateParticleDensity` |
| T6 | pressure solve | `solveIncompressibility` |
| T7 | transfer G2P | `transferVelocities(toGrid=false)` |
| T8 | colors | `updateParticleColors` + `updateCellColors` |
| T9 | render | GL draw + swap |
| T10 | D2H copy (CUDA only) | render readback |

### CUDA parallelization choices
- **Spatial hash (T2):** atomic histogram → Thrust `inclusive_scan` → atomic
  scatter. Separation sweeps are **double-buffered Jacobi-style** (each thread
  moves only itself by its own half-displacement) — race-free and deterministic.
- **Pressure solve (T6):** two interchangeable solvers, `--solver rb` (Red-Black
  Gauss-Seidel, default — closest to the CPU sweep) and `--solver jacobi`
  (delta-accumulation Jacobi). Both run race-free.
- **Reductions/scan:** Thrust (`inclusive_scan`, `transform_reduce`, `count`).
- **Constant memory:** grid scalars live in `__constant__`.

## Build

```sh
# CPU build  -> ./flip
make

# CUDA build -> ./flip_cuda   (set NVCC_ARCH to your GPU's compute capability)
make cuda NVCC_ARCH=sm_86
```

Find your GPU's compute capability with `nvidia-smi --query-gpu=compute_cap --format=csv`
(or the CUDA `deviceQuery` sample), then pass it as `sm_XX`.

## Run (interactive)

```sh
./flip       --no-vsync
./flip_cuda  --no-vsync --solver rb        # or --solver jacobi
```

Controls: **LMB drag** move obstacle · **SPACE/P** pause · **G** toggle grid ·
**R** reset · **Q/Esc** quit.

## Benchmark (assignment matrix)

Runs the fixed config (gravity on, separate-particles on, compensate-drift on,
`flipRatio=0.9`, static obstacle at (3,2)), discards 60 warm-up frames, then
averages per-stage timings over 600 steady-state frames — for resolutions
**50, 100, 150, 200**. Always run with VSync off.

```sh
./flip       --benchmark --no-vsync                 # -> bench_cpu.csv
./flip_cuda  --benchmark --no-vsync --solver rb     # -> bench_cuda.csv
./flip_cuda  --benchmark --no-vsync --solver jacobi # compare solvers
```

Add `--res N` to benchmark a single resolution. CSV columns:
`build,resolution,numParticles,numPressureIters,T1..T10,T_total` (ms/frame mean).

## Numerical validation

Dumps per-frame aggregate scalars (mean speed, total KE, centroid) for a
CPU-vs-CUDA diff:

```sh
./flip       --dump-stats --res 100 --frames 300    # -> stats_cpu.csv
./flip_cuda  --dump-stats --res 100 --frames 300    # -> stats_cuda.csv
```

Trajectories should stay visually identical; aggregate scalars track within a
small band (Gauss-Seidel→Red-Black/Jacobi reordering and float atomic ordering
cause expected small divergence).

## Hardware to report (for the writeup)

CPU (model + clock), GPU (model + compute capability + peak memory bandwidth
from datasheet), RAM, OS, NVIDIA driver version, CUDA toolkit version
(`nvcc --version`).
