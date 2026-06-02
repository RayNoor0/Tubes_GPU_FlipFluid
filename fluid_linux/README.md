# FLIP Fluid Simulator — CPU & CUDA

A 2D FLIP/PIC fluid simulation, Created to fulfill GPU coursework.
Two builds: a single-threaded **CPU** version (`flip`) and a **CUDA** GPU version (`flip_cuda`).
Both print a per-stage timing table so the pipeline can be compared stage by stage.

## Requirements

Runs on **Linux or WSL2**.

The CUDA build also needs the **CUDA toolkit** (`nvcc`) and an NVIDIA GPU.

## Build & run — CPU

```sh
make                # builds ./flip
./flip              # starts paused — press SPACE to run
```

## Build & run — CUDA

```sh
cd cuda
make                # builds ./flip_cuda
./flip_cuda
```

> On WSL2, CUDA–OpenGL interop is unsupported, so the GPU build automatically falls back to a
> device→host copy for rendering (the panel shows `Render: D2H copy`). On native Linux/Windows
> with the NVIDIA driver it auto-enables true interop (`Render: INTEROP (B1)`).

## Controls

| Key / action | Effect |
|---|---|
| **SPACE** or **P** | pause / resume |
| **left mouse drag** | move the obstacle |
| **G** | toggle the grid overlay |
| **R** | reset the scene |
| **Q** / **Esc** | quit |

The on-screen panel also has checkboxes/sliders (PIC↔FLIP ratio, grid resolution, gravity, etc.).
Once running, each build prints an averaged per-stage timing table to the terminal every 60 frames.

## Command-line options (both builds)

| Flag | Description |
|---|---|
| `--no-vsync` | uncap the frame rate (recommended for measuring) |
| `--bench` | run the fixed benchmark (resolutions 50/100/150/200, 60 warm-up + 600 measured frames, vsync off) and exit, printing a stage × resolution matrix of avg ms/frame |
| `--validate` | run a deterministic numeric pass (120 frames per resolution, no rendering) and print aggregate particle statistics, then exit |
| `-h`, `--help` | usage |

### Benchmark

```sh
./flip --bench               > cpu_bench.txt
cd cuda && ./flip_cuda --bench > cuda_bench.txt   # also prints GPU info
```

### Numerical validation (compare CPU vs CUDA)

```sh
./flip --validate                  > cpu_validate.txt
cd cuda && ./flip_cuda --validate  > cuda_validate.txt
# then diff the two — bulk quantities should agree to a few significant figures
```

