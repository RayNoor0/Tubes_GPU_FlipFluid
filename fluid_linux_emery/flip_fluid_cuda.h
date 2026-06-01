// CUDA port of the FLIP fluid simulator. Mirrors flipcpu::FlipFluid stage for
// stage, but every pipeline step runs as one or more CUDA kernels operating on
// device-resident SoA buffers.
//
// Scene setup still happens on the host (a flipcpu::FlipFluid is used as the
// staging object): positions/velocities/solid-mask are computed on the CPU then
// uploaded once via uploadStateFrom(). For rendering (no GL interop in this
// build) the render-relevant arrays are copied back D2H each frame
// (downloadForRender, measured as T10_d2h).

#pragma once

#include "flip_fluid.h"   // host staging object + FLUID/AIR/SOLID constants
#include "bench.h"

#include <cuda_runtime.h>

namespace flipgpu {

enum SolverType { SOLVER_RED_BLACK = 0, SOLVER_JACOBI = 1 };

class FlipFluidCuda {
public:
    // Geometry/scalars — computed identically to flipcpu::FlipFluid so the two
    // builds share the exact same grid and particle layout.
    float density;
    int   fNumX, fNumY, fNumCells;
    float h, fInvSpacing;
    int   maxParticles;
    float particleRadius, pInvSpacing;
    int   pNumX, pNumY, pNumCells;
    int   numParticles;
    float particleRestDensity;

    SolverType solver = SOLVER_RED_BLACK;
    flipbench::StageTimers* timers = nullptr;

    FlipFluidCuda(float density, float width, float height,
                  float spacing, float particle_radius, int max_particles);
    ~FlipFluidCuda();

    // Upload the full initial state from a host FlipFluid.
    void uploadStateFrom(const flipcpu::FlipFluid& f);
    // Re-upload only the solid mask (after the obstacle is re-carved on host).
    void uploadSolid(const flipcpu::FlipFluid& f);

    void simulate(float dt, float gravity, float flipRatio,
                  int numPressureIters, int numParticleIters,
                  float overRelaxation, bool compensateDrift,
                  bool separateParticles,
                  float obstacleX, float obstacleY, float obstacleRadius,
                  float obstacleVelX, float obstacleVelY,
                  int numSubSteps = 1);

    // Copy particle pos/color + cell color back into a host FlipFluid so the
    // legacy-GL draw path can render. Measured as T10_d2h.
    void downloadForRender(flipcpu::FlipFluid& host);
    // Copy particle positions + velocities back (for validation statistics).
    void downloadParticles(flipcpu::FlipFluid& host);

private:
    // Grid buffers (size fNumCells)
    float *d_u, *d_v, *d_du, *d_dv, *d_prevU, *d_prevV, *d_p, *d_s;
    int   *d_cellType;
    float *d_cellColor;        // 3 * fNumCells
    float *d_particleDensity;

    // Particle SoA (size maxParticles)
    float *d_posX, *d_posY, *d_velX, *d_velY;
    float *d_colR, *d_colG, *d_colB;
    // Double buffers for the pushParticlesApart separation sweeps.
    float *d_posX2, *d_posY2, *d_colR2, *d_colG2, *d_colB2;

    // Spatial hash
    int *d_numCellParticles;     // pNumCells
    int *d_firstCellParticle;    // pNumCells + 1
    int *d_cellParticleIds;      // maxParticles

    cudaEvent_t evStart, evStop;

    void allocate();
    void freeAll();
    void uploadParams();

    // Stage groups (host launchers).
    void pushParticlesApart(int numIters);
    void transferToGrid();
    void densityStep();
    void solvePressure(int numIters, float dt, float overRelaxation,
                       bool compensateDrift);
    void transferFromGrid(float flipRatio);
    void colorStep();
    float computeRestDensityGPU();
};

} // namespace flipgpu
