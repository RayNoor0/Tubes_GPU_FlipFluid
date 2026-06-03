// CUDA port of the FLIP/PIC fluid solver (mirrors flipcpu::FlipFluid).
//
// The simulation runs entirely on the GPU. Particle positions and colors live in
// OpenGL VBOs registered with CUDA (cudaGraphicsGLRegisterBuffer), so kernels write
// straight into the buffers OpenGL renders from — no cudaMemcpyDeviceToHost on the
// render path (bonus B1). Per-stage timing uses cudaEvent_t pairs; see simulate().
//
// Grid layout matches the CPU version exactly: index = i * fNumY + j (n = fNumY),
// staggered MAC grid. Numerical results diverge from the CPU only where parallelism
// forces it: the pressure solve uses red-black Gauss-Seidel (same overRelaxation and
// iteration count as the CPU, only the update ordering differs) and particle
// separation is Jacobi-style (documented at the call sites).

#pragma once

#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <cstdio>
#include <cstdlib>

#include "../stage_timing.h"

namespace flipcuda {

constexpr int U_FIELD = 0;
constexpr int V_FIELD = 1;

constexpr int FLUID_CELL = 0;
constexpr int AIR_CELL   = 1;
constexpr int SOLID_CELL = 2;

constexpr int BLOCK = 256;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(_err));                             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

class FlipFluidCuda {
public:
    // --- grid (mirror of CPU fields) ---
    float density;
    int   fNumX, fNumY, fNumCells;
    float h, fInvSpacing;

    // --- particles ---
    int   maxParticles;
    int   numParticles = 0;
    float particleRadius;
    float pInvSpacing;
    int   pNumX, pNumY, pNumCells;
    float particleRestDensity = 0.0f;

    // --- device grid buffers (fNumCells each unless noted) ---
    float* d_u = nullptr;  float* d_v = nullptr;
    float* d_du = nullptr; float* d_dv = nullptr;
    float* d_prevU = nullptr; float* d_prevV = nullptr;
    float* d_p = nullptr;  float* d_s = nullptr;
    int*   d_cellType = nullptr;
    float* d_particleDensity = nullptr;
    float* d_cellColor = nullptr;          // 3 * fNumCells

    // --- device particle buffers ---
    float2* d_vel = nullptr;               // maxParticles
    float2* d_posScratch = nullptr;        // maxParticles (Jacobi separation snapshot)
    float3* d_colScratch = nullptr;        // maxParticles

    // Owned position/color buffers used in NO-INTEROP mode (WSL2 fallback). In interop
    // mode the kernels write the mapped VBO pointers instead and these stay idle.
    float2* d_pos = nullptr;               // maxParticles
    float3* d_col = nullptr;               // maxParticles

    // Host readback for the no-interop render path (and frame-0 seed). 2*/3* maxParticles.
    std::vector<float> hostPos;
    std::vector<float> hostCol;
    bool interopEnabled = false;

    // --- spatial hash (particle separation) ---
    int* d_numCellParticles = nullptr;     // pNumCells
    int* d_firstCellParticle = nullptr;    // pNumCells + 1
    int* d_cellParticleIds = nullptr;      // maxParticles
    int* d_cellCursor = nullptr;           // pNumCells (scatter cursor)

    // --- reduction scratch (computeRestDensity) ---
    float* d_sum = nullptr;                // 1
    int*   d_count = nullptr;              // 1

    // --- OpenGL interop (B1) ---
    GLuint posVBO = 0, colVBO = 0;
    cudaGraphicsResource* posRes = nullptr;
    cudaGraphicsResource* colRes = nullptr;

    // --- timing ---
    fliptiming::StageStats timing;
    cudaEvent_t evStart[fliptiming::STAGE_COUNT];
    cudaEvent_t evStop [fliptiming::STAGE_COUNT];

    FlipFluidCuda(float density, float width, float height,
                  float spacing, float particle_radius, int max_particles);
    ~FlipFluidCuda();

    // Try to register externally-created VBOs with CUDA (GL context must be current).
    // Returns true on success (interop enabled). On failure (e.g. WSL2, where GL-CUDA
    // interop is unsupported) it clears the CUDA error and returns false, leaving the
    // object in no-interop mode — the caller should render from hostPositions()/Colors().
    bool tryRegisterVBOs(GLuint pos, GLuint col);
    bool usingInterop() const { return interopEnabled; }

    // Upload host-side scene data.
    void uploadParticles(const std::vector<float>& posXY,
                         const std::vector<float>& colRGB); // -> d_pos/d_col + host seed
    void uploadSolid(const std::vector<float>& s);   // -> d_s
    void resetVelocities();                           // d_vel = 0

    // No-interop render source (valid after uploadParticles / each simulate()).
    const std::vector<float>& hostPositions() const { return hostPos; }
    const std::vector<float>& hostColors() const { return hostCol; }

    // Copy current particle positions (float2) and velocities (float2) to host,
    // interleaved as [x0,y0,x1,y1,...]. Used by the --validate mode; works in both
    // interop (maps the VBO) and no-interop modes.
    void downloadParticles(std::vector<float>& posXY, std::vector<float>& velXY);

    // Obstacle carve (mirror of carveObstacle): writes d_s, d_u, d_v.
    void carveObstacle(float x, float y, float r, float vx, float vy);

    // One frame: maps the VBOs, runs the staged pipeline, unmaps. Fills timing
    // for T1..T8 and T10. (T9_render / T_total are timed by the host loop.)
    void simulate(float dt, float gravity, float flipRatio,
                  int numPressureIters, int numParticleIters,
                  float overRelaxation, bool compensateDrift,
                  bool separateParticles,
                  float obstacleX, float obstacleY, float obstacleRadius,
                  float obstacleVelX, float obstacleVelY,
                  int numSubSteps);

    // For the optional grid overlay (debug, default off): copy cellColor to host.
    void downloadCellColors(std::vector<float>& out);

    // Recompute the cell colors once while paused (so frame 0 looks right).
    void updateCellColorsOnce();

private:
    bool restDensityComputed = false;

    // Map/unmap the interop VBOs; valid only between map and unmap.
    void mapVBOs(float2** outPos, float3** outCol);
    void unmapVBOs();

    // Stage helpers (each launches one or more kernels on the default stream).
    void k_integrate(float2* pos, float dt, float gravity);
    void k_collisions(float2* pos, float ox, float oy, float orad,
                      float ovx, float ovy);
    void transferToGrid(float2* pos);                 // T4
    void updateDensity(float2* pos);                  // T5
    void computeRestDensityIfNeeded();
    void solvePressure(int numIters, float dt, float overRelaxation,
                       bool compensateDrift);         // T6 (red-black GS)
    void transferToParticles(float2* pos, float flipRatio); // T7
    void updateColors(float2* pos, float3* col);      // T8
    void pushApart(float2* pos, float3* col, int numIters); // T2
};

} // namespace flipcuda
