// CUDA implementation of the FLIP pipeline. See flip_fluid_cuda.h.
//
// Parallelization strategy per stage (justification for the report):
//   T1 integrate           : 1 thread / particle.
//   T2 pushParticlesApart  : atomic histogram -> Thrust inclusive_scan ->
//                            atomic scatter fill -> double-buffered Jacobi-style
//                            separation (each thread moves only itself by its
//                            own half-displacement; race-free, deterministic).
//   T3 collisions          : 1 thread / particle.
//   T4 transfer-to-grid    : per-cell savePrev + classify-init, per-particle
//                            classify-scatter + p2g (atomicAdd), per-cell
//                            normalize + restore-solid.
//   T5 density             : per-particle atomicAdd scatter; rest density via
//                            Thrust transform_reduce/count (first frame only).
//   T6 pressure            : Red-Black Gauss-Seidel (2 race-free kernels/iter)
//                            OR Jacobi (delta accumulation via atomics + apply).
//   T7 transfer-from-grid  : 1 thread / particle, read-only grid gather.
//   T8 colors              : per-particle + per-cell.

#include "flip_fluid_cuda.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <utility>      // std::swap

namespace flipgpu {

using flipcpu::FLUID_CELL;
using flipcpu::AIR_CELL;
using flipcpu::SOLID_CELL;

static const int BS = 256;
static inline int pgrid(int n) { return (n + BS - 1) / BS; }

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "[cuda] %s:%d %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(_e));                              \
        }                                                                      \
    } while (0)

// Time a stage group into the optional StageTimers via cudaEvents. Variadic so
// kernel-launch commas (inside <<<grid, block>>>) survive macro expansion.
#define CUTIME(STAGE, ...)                                                     \
    do {                                                                       \
        if (timers) {                                                          \
            cudaEventRecord(evStart);                                          \
            __VA_ARGS__;                                                       \
            cudaEventRecord(evStop);                                           \
            cudaEventSynchronize(evStop);                                      \
            float _ms = 0.0f;                                                  \
            cudaEventElapsedTime(&_ms, evStart, evStop);                       \
            timers->add(flipbench::STAGE, _ms);                                \
        } else {                                                               \
            __VA_ARGS__;                                                       \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constant grid parameters (read by every kernel for index math).
// ---------------------------------------------------------------------------
struct GpuParams {
    int   fNumX, fNumY, n;     // n == fNumY (grid stride)
    float h, fInvSpacing, h2;  // h2 == 0.5 * h
    int   pNumX, pNumY;
    float pInvSpacing;
    float particleRadius;
    float density;
};
__constant__ GpuParams gp;

__device__ __forceinline__ float dclampf(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}
__device__ __forceinline__ int dclampi(int x, int lo, int hi) {
    return max(lo, min(hi, x));
}

// Functor for the rest-density reduction: density of FLUID cells, 0 elsewhere.
// File-scope so nvcc accepts it as a Thrust template argument; templated on the
// tuple type so it accepts the zip_iterator's reference type directly.
struct MaskSum {
    template <class Tuple>
    __host__ __device__ float operator()(const Tuple& t) const {
        return thrust::get<1>(t) == FLUID_CELL ? (float)thrust::get<0>(t) : 0.0f;
    }
};
// Bounds-safe cellType read: out-of-domain neighbours count as AIR. (The CPU
// reference reads slightly out of bounds at the extreme boundary cell; treating
// those as AIR is the safe, well-defined equivalent on the GPU.)
__device__ __forceinline__ int ctAt(const int* cellType, int idx, int nCells) {
    return (idx < 0 || idx >= nCells) ? AIR_CELL : cellType[idx];
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void k_integrate(float* posX, float* posY, float* velX, float* velY,
                            int numParticles, float dt, float gravity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    velY[i] += dt * gravity;
    posX[i] += velX[i] * dt;
    posY[i] += velY[i] * dt;
}

__global__ void k_countParticles(const float* posX, const float* posY,
                                  int* numCellParticles, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int xi = dclampi((int)floorf(posX[i] * gp.pInvSpacing), 0, gp.pNumX - 1);
    int yi = dclampi((int)floorf(posY[i] * gp.pInvSpacing), 0, gp.pNumY - 1);
    atomicAdd(&numCellParticles[xi * gp.pNumY + yi], 1);
}

__global__ void k_fillParticles(const float* posX, const float* posY,
                                 int* firstCellParticle, int* cellParticleIds,
                                 int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int xi = dclampi((int)floorf(posX[i] * gp.pInvSpacing), 0, gp.pNumX - 1);
    int yi = dclampi((int)floorf(posY[i] * gp.pInvSpacing), 0, gp.pNumY - 1);
    int cell = xi * gp.pNumY + yi;
    int slot = atomicSub(&firstCellParticle[cell], 1) - 1;
    cellParticleIds[slot] = i;
}

// Double-buffered Jacobi-style separation: thread i reads neighbours from the
// "in" buffers, accumulates only its own half of each pairwise push, and writes
// to the "out" buffers.
__global__ void k_pushIteration(const float* pinX, const float* pinY,
                                 const float* cinR, const float* cinG, const float* cinB,
                                 float* poutX, float* poutY,
                                 float* coutR, float* coutG, float* coutB,
                                 const int* firstCellParticle,
                                 const int* cellParticleIds, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const float colorDiffusionCoeff = 0.001f;
    float px = pinX[i], py = pinY[i];
    float minDist  = 2.0f * gp.particleRadius;
    float minDist2 = minDist * minDist;

    int pxi = (int)floorf(px * gp.pInvSpacing);
    int pyi = (int)floorf(py * gp.pInvSpacing);
    int x0 = max(pxi - 1, 0), y0 = max(pyi - 1, 0);
    int x1 = min(pxi + 1, gp.pNumX - 1), y1 = min(pyi + 1, gp.pNumY - 1);

    float dispx = 0.0f, dispy = 0.0f;
    float c0r = cinR[i], c0g = cinG[i], c0b = cinB[i];
    float cr = c0r, cg = c0g, cb = c0b;

    for (int xi = x0; xi <= x1; ++xi) {
        for (int yi = y0; yi <= y1; ++yi) {
            int cell = xi * gp.pNumY + yi;
            int firstI = firstCellParticle[cell];
            int lastI  = firstCellParticle[cell + 1];
            for (int j = firstI; j < lastI; ++j) {
                int idn = cellParticleIds[j];
                if (idn == i) continue;
                float qx = pinX[idn], qy = pinY[idn];
                float dx = qx - px, dy = qy - py;
                float d2 = dx * dx + dy * dy;
                if (d2 > minDist2 || d2 == 0.0f) continue;
                float d = sqrtf(d2);
                float sFac = 0.5f * (minDist - d) / d;
                dispx -= dx * sFac;     // this particle's own half
                dispy -= dy * sFac;
                float c1r = cinR[idn], c1g = cinG[idn], c1b = cinB[idn];
                cr += ((c0r + c1r) * 0.5f - c0r) * colorDiffusionCoeff;
                cg += ((c0g + c1g) * 0.5f - c0g) * colorDiffusionCoeff;
                cb += ((c0b + c1b) * 0.5f - c0b) * colorDiffusionCoeff;
            }
        }
    }
    poutX[i] = px + dispx;
    poutY[i] = py + dispy;
    coutR[i] = cr; coutG[i] = cg; coutB[i] = cb;
}

__global__ void k_collide(float* posX, float* posY, float* velX, float* velY,
                          int numParticles, float obstacleX, float obstacleY,
                          float obstacleRadius, float obstacleVelX, float obstacleVelY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float hh = 1.0f / gp.fInvSpacing;
    float r  = gp.particleRadius;
    float minDist  = obstacleRadius + r;
    float minDist2 = minDist * minDist;
    float minX = hh + r, maxX = (gp.fNumX - 1) * hh - r;
    float minY = hh + r, maxY = (gp.fNumY - 1) * hh - r;

    float x = posX[i], y = posY[i];
    float dx = x - obstacleX, dy = y - obstacleY;
    if (dx * dx + dy * dy < minDist2) {
        velX[i] = obstacleVelX;
        velY[i] = obstacleVelY;
    }
    if (x < minX) { x = minX; velX[i] = 0.0f; }
    if (x > maxX) { x = maxX; velX[i] = 0.0f; }
    if (y < minY) { y = minY; velY[i] = 0.0f; }
    if (y > maxY) { y = maxY; velY[i] = 0.0f; }
    posX[i] = x; posY[i] = y;
}

__global__ void k_particleDensity(const float* posX, const float* posY,
                                  float* particleDensity, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int   n  = gp.n;
    float hh = gp.h, h1 = gp.fInvSpacing, h2 = gp.h2;

    float x = dclampf(posX[i], hh, (gp.fNumX - 1) * hh);
    float y = dclampf(posY[i], hh, (gp.fNumY - 1) * hh);

    int   x0 = (int)floorf((x - h2) * h1);
    float tx = ((x - h2) - x0 * hh) * h1;
    int   x1 = min(x0 + 1, gp.fNumX - 2);
    int   y0 = (int)floorf((y - h2) * h1);
    float ty = ((y - h2) - y0 * hh) * h1;
    int   y1 = min(y0 + 1, gp.fNumY - 2);
    float sx = 1.0f - tx, sy = 1.0f - ty;

    if (x0 < gp.fNumX && y0 < gp.fNumY) atomicAdd(&particleDensity[x0 * n + y0], sx * sy);
    if (x1 < gp.fNumX && y0 < gp.fNumY) atomicAdd(&particleDensity[x1 * n + y0], tx * sy);
    if (x1 < gp.fNumX && y1 < gp.fNumY) atomicAdd(&particleDensity[x1 * n + y1], tx * ty);
    if (x0 < gp.fNumX && y1 < gp.fNumY) atomicAdd(&particleDensity[x0 * n + y1], sx * ty);
}

__global__ void k_savePrev(float* u, float* v, float* du, float* dv,
                           float* prevU, float* prevV, int fNumCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    prevU[i] = u[i]; prevV[i] = v[i];
    du[i] = 0.0f; dv[i] = 0.0f;
    u[i] = 0.0f; v[i] = 0.0f;
}

__global__ void k_classifyInit(const float* s, int* cellType, int fNumCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    cellType[i] = (s[i] == 0.0f) ? SOLID_CELL : AIR_CELL;
}

__global__ void k_classifyScatter(const float* posX, const float* posY,
                                  int* cellType, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int xi = dclampi((int)floorf(posX[i] * gp.fInvSpacing), 0, gp.fNumX - 1);
    int yi = dclampi((int)floorf(posY[i] * gp.fInvSpacing), 0, gp.fNumY - 1);
    int cell = xi * gp.fNumY + yi;
    // Benign race: all writers store the same value, SOLID is never overwritten.
    if (cellType[cell] == AIR_CELL) cellType[cell] = FLUID_CELL;
}

__global__ void k_p2g(int component, const float* posX, const float* posY,
                      const float* pvel, float* fld, float* fldD, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int   n  = gp.n;
    float hh = gp.h, h1 = gp.fInvSpacing, h2 = gp.h2;
    float dxOff = (component == 0) ? 0.0f : h2;
    float dyOff = (component == 0) ? h2   : 0.0f;

    float x = dclampf(posX[i], hh, (gp.fNumX - 1) * hh);
    float y = dclampf(posY[i], hh, (gp.fNumY - 1) * hh);

    int   x0 = min((int)floorf((x - dxOff) * h1), gp.fNumX - 2);
    float tx = ((x - dxOff) - x0 * hh) * h1;
    int   x1 = min(x0 + 1, gp.fNumX - 2);
    int   y0 = min((int)floorf((y - dyOff) * h1), gp.fNumY - 2);
    float ty = ((y - dyOff) - y0 * hh) * h1;
    int   y1 = min(y0 + 1, gp.fNumY - 2);
    float sx = 1.0f - tx, sy = 1.0f - ty;

    float d0 = sx * sy, d1 = tx * sy, d2 = tx * ty, d3 = sx * ty;
    int nr0 = x0 * n + y0, nr1 = x1 * n + y0, nr2 = x1 * n + y1, nr3 = x0 * n + y1;
    float pv = pvel[i];

    atomicAdd(&fld[nr0], pv * d0); atomicAdd(&fldD[nr0], d0);
    atomicAdd(&fld[nr1], pv * d1); atomicAdd(&fldD[nr1], d1);
    atomicAdd(&fld[nr2], pv * d2); atomicAdd(&fldD[nr2], d2);
    atomicAdd(&fld[nr3], pv * d3); atomicAdd(&fldD[nr3], d3);
}

__global__ void k_p2gNormalize(float* fld, const float* fldD, int fNumCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    if (fldD[i] > 0.0f) fld[i] /= fldD[i];
}

__global__ void k_restoreSolid(float* u, float* v, const float* prevU,
                               const float* prevV, const int* cellType, int fNumCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fNumCells) return;
    int n = gp.n;
    int i = idx / n, j = idx % n;
    bool solid = (cellType[idx] == SOLID_CELL);
    if (solid || (i > 0 && cellType[(i - 1) * n + j] == SOLID_CELL)) u[idx] = prevU[idx];
    if (solid || (j > 0 && cellType[idx - 1] == SOLID_CELL))         v[idx] = prevV[idx];
}

__global__ void k_g2p(int component, const float* posX, const float* posY,
                      float* pvel, const float* fld, const float* pfld,
                      const int* cellType, float flipRatio,
                      int numParticles, int fNumCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int   n  = gp.n;
    float hh = gp.h, h1 = gp.fInvSpacing, h2 = gp.h2;
    float dxOff  = (component == 0) ? 0.0f : h2;
    float dyOff  = (component == 0) ? h2   : 0.0f;
    int   offset = (component == 0) ? n : 1;

    float x = dclampf(posX[i], hh, (gp.fNumX - 1) * hh);
    float y = dclampf(posY[i], hh, (gp.fNumY - 1) * hh);

    int   x0 = min((int)floorf((x - dxOff) * h1), gp.fNumX - 2);
    float tx = ((x - dxOff) - x0 * hh) * h1;
    int   x1 = min(x0 + 1, gp.fNumX - 2);
    int   y0 = min((int)floorf((y - dyOff) * h1), gp.fNumY - 2);
    float ty = ((y - dyOff) - y0 * hh) * h1;
    int   y1 = min(y0 + 1, gp.fNumY - 2);
    float sx = 1.0f - tx, sy = 1.0f - ty;

    float d0 = sx * sy, d1 = tx * sy, d2 = tx * ty, d3 = sx * ty;
    int nr0 = x0 * n + y0, nr1 = x1 * n + y0, nr2 = x1 * n + y1, nr3 = x0 * n + y1;

    float valid0 = (ctAt(cellType, nr0, fNumCells) != AIR_CELL || ctAt(cellType, nr0 - offset, fNumCells) != AIR_CELL) ? 1.0f : 0.0f;
    float valid1 = (ctAt(cellType, nr1, fNumCells) != AIR_CELL || ctAt(cellType, nr1 - offset, fNumCells) != AIR_CELL) ? 1.0f : 0.0f;
    float valid2 = (ctAt(cellType, nr2, fNumCells) != AIR_CELL || ctAt(cellType, nr2 - offset, fNumCells) != AIR_CELL) ? 1.0f : 0.0f;
    float valid3 = (ctAt(cellType, nr3, fNumCells) != AIR_CELL || ctAt(cellType, nr3 - offset, fNumCells) != AIR_CELL) ? 1.0f : 0.0f;

    float v_old = pvel[i];
    float d = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;
    if (d > 0.0f) {
        float f0 = fld[nr0],  f1 = fld[nr1],  f2 = fld[nr2],  f3 = fld[nr3];
        float pf0 = pfld[nr0], pf1 = pfld[nr1], pf2 = pfld[nr2], pf3 = pfld[nr3];
        float picV = (valid0 * d0 * f0 + valid1 * d1 * f1
                    + valid2 * d2 * f2 + valid3 * d3 * f3) / d;
        float corr = (valid0 * d0 * (f0 - pf0) + valid1 * d1 * (f1 - pf1)
                    + valid2 * d2 * (f2 - pf2) + valid3 * d3 * (f3 - pf3)) / d;
        float flipV = v_old + corr;
        pvel[i] = (1.0f - flipRatio) * picV + flipRatio * flipV;
    }
}

__global__ void k_solveInit(const float* u, const float* v, float* p,
                            float* prevU, float* prevV, int fNumCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    p[i] = 0.0f;
    prevU[i] = u[i];
    prevV[i] = v[i];
}

// Red-Black Gauss-Seidel sweep: one parity ("color") per launch. No two
// same-parity fluid cells write the same MAC face, so this is race-free.
__global__ void k_solveRB(int color, float* u, float* v, float* p,
                          const float* s, const int* cellType,
                          const float* particleDensity, float overRelaxation,
                          float cp, float rest, int compensateDrift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = gp.n;
    int i = idx / n, j = idx % n;
    if (i < 1 || i >= gp.fNumX - 1 || j < 1 || j >= gp.fNumY - 1) return;
    if (((i + j) & 1) != color) return;
    if (cellType[idx] != FLUID_CELL) return;

    int center = idx;
    int left = (i - 1) * n + j, right = (i + 1) * n + j;
    int bottom = idx - 1, top = idx + 1;

    float sx0 = s[left], sx1 = s[right], sy0 = s[bottom], sy1 = s[top];
    float sSum = sx0 + sx1 + sy0 + sy1;
    if (sSum == 0.0f) return;

    float div = u[right] - u[center] + v[top] - v[center];
    if (rest > 0.0f && compensateDrift) {
        float compression = particleDensity[center] - rest;
        if (compression > 0.0f) div -= compression;   // k == 1
    }
    float pVal = -div / sSum * overRelaxation;
    p[center] += cp * pVal;
    u[center] -= sx0 * pVal;
    u[right]  += sx1 * pVal;
    v[center] -= sy0 * pVal;
    v[top]    += sy1 * pVal;
}

// Jacobi: every fluid cell computes its correction from the frozen u/v and
// scatters it into the delta buffers (du/dv) via atomics; a second kernel adds
// the deltas back. Converges to the same projection as GS, but slower.
__global__ void k_jacobiAccum(const float* u, const float* v, float* p,
                              float* du, float* dv, const float* s,
                              const int* cellType, const float* particleDensity,
                              float overRelaxation, float cp, float rest,
                              int compensateDrift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = gp.n;
    int i = idx / n, j = idx % n;
    if (i < 1 || i >= gp.fNumX - 1 || j < 1 || j >= gp.fNumY - 1) return;
    if (cellType[idx] != FLUID_CELL) return;

    int center = idx;
    int left = (i - 1) * n + j, right = (i + 1) * n + j;
    int bottom = idx - 1, top = idx + 1;

    float sx0 = s[left], sx1 = s[right], sy0 = s[bottom], sy1 = s[top];
    float sSum = sx0 + sx1 + sy0 + sy1;
    if (sSum == 0.0f) return;

    float div = u[right] - u[center] + v[top] - v[center];
    if (rest > 0.0f && compensateDrift) {
        float compression = particleDensity[center] - rest;
        if (compression > 0.0f) div -= compression;
    }
    float pVal = -div / sSum * overRelaxation;
    p[center] += cp * pVal;
    atomicAdd(&du[center], -sx0 * pVal);
    atomicAdd(&du[right],   sx1 * pVal);
    atomicAdd(&dv[center], -sy0 * pVal);
    atomicAdd(&dv[top],     sy1 * pVal);
}

__global__ void k_jacobiApply(float* u, float* v, const float* du,
                              const float* dv, int fNumCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    u[i] += du[i];
    v[i] += dv[i];
}

__global__ void k_updateParticleColors(const float* posX, const float* posY,
                                       float* colR, float* colG, float* colB,
                                       const float* particleDensity,
                                       float restDensity, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    const float sStep = 0.01f;
    float h1 = gp.fInvSpacing;

    colR[i] = dclampf(colR[i] - sStep, 0.0f, 1.0f);
    colG[i] = dclampf(colG[i] - sStep, 0.0f, 1.0f);
    colB[i] = dclampf(colB[i] + sStep, 0.0f, 1.0f);

    int xi = dclampi((int)floorf(posX[i] * h1), 1, gp.fNumX - 1);
    int yi = dclampi((int)floorf(posY[i] * h1), 1, gp.fNumY - 1);
    int cell = xi * gp.fNumY + yi;
    if (restDensity > 0.0f) {
        float relDensity = particleDensity[cell] / restDensity;
        if (relDensity < 0.7f) {
            colR[i] = 0.8f; colG[i] = 0.8f; colB[i] = 1.0f;
        }
    }
}

__device__ void d_setSciColor(float* cellColor, int cellNr, float val,
                              float minVal, float maxVal) {
    val = fminf(fmaxf(val, minVal), maxVal - 0.0001f);
    float d = maxVal - minVal;
    val = (d == 0.0f) ? 0.5f : (val - minVal) / d;
    float m = 0.25f;
    int num = (int)floorf(val / m);
    float sLoc = (val - num * m) / m;
    float r = 0, g = 0, b = 0;
    switch (num) {
        case 0: r = 0.0f;  g = sLoc;        b = 1.0f;        break;
        case 1: r = 0.0f;  g = 1.0f;        b = 1.0f - sLoc; break;
        case 2: r = sLoc;  g = 1.0f;        b = 0.0f;        break;
        default: r = 1.0f; g = 1.0f - sLoc; b = 0.0f;        break;
    }
    cellColor[3 * cellNr]     = r;
    cellColor[3 * cellNr + 1] = g;
    cellColor[3 * cellNr + 2] = b;
}

__global__ void k_updateCellColors(const int* cellType, const float* particleDensity,
                                   float* cellColor, float restDensity, int fNumCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    if (cellType[i] == SOLID_CELL) {
        cellColor[3 * i] = 0.5f; cellColor[3 * i + 1] = 0.5f; cellColor[3 * i + 2] = 0.5f;
    } else if (cellType[i] == FLUID_CELL) {
        float d = particleDensity[i];
        if (restDensity > 0.0f) d /= restDensity;
        d_setSciColor(cellColor, i, d, 0.0f, 2.0f);
    } else {
        cellColor[3 * i] = 0.0f; cellColor[3 * i + 1] = 0.0f; cellColor[3 * i + 2] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// FlipFluidCuda host code
// ---------------------------------------------------------------------------

FlipFluidCuda::FlipFluidCuda(float density_, float width, float height,
                             float spacing, float particle_radius, int max_particles) {
    density = density_;
    fNumX = (int)std::floor(width / spacing) + 1;
    fNumY = (int)std::floor(height / spacing) + 1;
    h = std::max(width / fNumX, height / fNumY);
    fInvSpacing = 1.0f / h;
    fNumCells = fNumX * fNumY;

    maxParticles = max_particles;
    particleRadius = particle_radius;
    pInvSpacing = 1.0f / (2.2f * particleRadius);
    pNumX = (int)std::floor(width * pInvSpacing) + 1;
    pNumY = (int)std::floor(height * pInvSpacing) + 1;
    pNumCells = pNumX * pNumY;

    numParticles = 0;
    particleRestDensity = 0.0f;

    allocate();
    uploadParams();
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
}

FlipFluidCuda::~FlipFluidCuda() {
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    freeAll();
}

void FlipFluidCuda::allocate() {
    size_t fc  = (size_t)fNumCells;
    size_t mp  = (size_t)maxParticles;
    size_t pc  = (size_t)pNumCells;
    auto fbytes = [](size_t n) { return n * sizeof(float); };
    auto ibytes = [](size_t n) { return n * sizeof(int); };

    CUDA_CHECK(cudaMalloc(&d_u, fbytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_v, fbytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_du, fbytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_dv, fbytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_prevU, fbytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_prevV, fbytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_p, fbytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_s, fbytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_cellType, ibytes(fc)));
    CUDA_CHECK(cudaMalloc(&d_cellColor, fbytes(3 * fc)));
    CUDA_CHECK(cudaMalloc(&d_particleDensity, fbytes(fc)));

    CUDA_CHECK(cudaMalloc(&d_posX, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_posY, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_velX, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_velY, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_colR, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_colG, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_colB, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_posX2, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_posY2, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_colR2, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_colG2, fbytes(mp)));
    CUDA_CHECK(cudaMalloc(&d_colB2, fbytes(mp)));

    CUDA_CHECK(cudaMalloc(&d_numCellParticles, ibytes(pc)));
    CUDA_CHECK(cudaMalloc(&d_firstCellParticle, ibytes(pc + 1)));
    CUDA_CHECK(cudaMalloc(&d_cellParticleIds, ibytes(mp)));
}

void FlipFluidCuda::freeAll() {
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_du); cudaFree(d_dv);
    cudaFree(d_prevU); cudaFree(d_prevV); cudaFree(d_p); cudaFree(d_s);
    cudaFree(d_cellType); cudaFree(d_cellColor); cudaFree(d_particleDensity);
    cudaFree(d_posX); cudaFree(d_posY); cudaFree(d_velX); cudaFree(d_velY);
    cudaFree(d_colR); cudaFree(d_colG); cudaFree(d_colB);
    cudaFree(d_posX2); cudaFree(d_posY2);
    cudaFree(d_colR2); cudaFree(d_colG2); cudaFree(d_colB2);
    cudaFree(d_numCellParticles); cudaFree(d_firstCellParticle); cudaFree(d_cellParticleIds);
}

void FlipFluidCuda::uploadParams() {
    GpuParams hp;
    hp.fNumX = fNumX; hp.fNumY = fNumY; hp.n = fNumY;
    hp.h = h; hp.fInvSpacing = fInvSpacing; hp.h2 = 0.5f * h;
    hp.pNumX = pNumX; hp.pNumY = pNumY; hp.pInvSpacing = pInvSpacing;
    hp.particleRadius = particleRadius; hp.density = density;
    CUDA_CHECK(cudaMemcpyToSymbol(gp, &hp, sizeof(GpuParams)));
}

void FlipFluidCuda::uploadStateFrom(const flipcpu::FlipFluid& f) {
    numParticles = f.numParticles;
    particleRestDensity = f.particleRestDensity;
    uploadParams();

    size_t fc = (size_t)fNumCells * sizeof(float);
    size_t np = (size_t)numParticles * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_s, f.s.data(), fc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, f.u.data(), fc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, f.v.data(), fc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posX, f.particlePosX.data(), np, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, f.particlePosY.data(), np, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, f.particleVelX.data(), np, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, f.particleVelY.data(), np, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colR, f.particleColorR.data(), np, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colG, f.particleColorG.data(), np, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colB, f.particleColorB.data(), np, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_particleDensity, 0, (size_t)fNumCells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cellColor, 0, (size_t)3 * fNumCells * sizeof(float)));
}

void FlipFluidCuda::uploadSolid(const flipcpu::FlipFluid& f) {
    CUDA_CHECK(cudaMemcpy(d_s, f.s.data(), (size_t)fNumCells * sizeof(float),
                          cudaMemcpyHostToDevice));
}

float FlipFluidCuda::computeRestDensityGPU() {
    thrust::device_ptr<float> dens(d_particleDensity);
    thrust::device_ptr<int>   ct(d_cellType);
    auto zb = thrust::make_zip_iterator(thrust::make_tuple(dens, ct));
    auto ze = thrust::make_zip_iterator(thrust::make_tuple(dens + fNumCells, ct + fNumCells));
    float sum = thrust::transform_reduce(zb, ze, MaskSum(), 0.0f, thrust::plus<float>());
    int   cnt = (int)thrust::count(ct, ct + fNumCells, FLUID_CELL);
    return (cnt > 0) ? sum / cnt : 0.0f;
}

void FlipFluidCuda::pushParticlesApart(int numIters) {
    CUDA_CHECK(cudaMemset(d_numCellParticles, 0, (size_t)pNumCells * sizeof(int)));
    k_countParticles<<<pgrid(numParticles), BS>>>(d_posX, d_posY, d_numCellParticles, numParticles);

    thrust::device_ptr<int> cnt(d_numCellParticles);
    thrust::device_ptr<int> first(d_firstCellParticle);
    thrust::inclusive_scan(cnt, cnt + pNumCells, first);
    // firstCellParticle[pNumCells] = total (== inclusive sum of last cell).
    CUDA_CHECK(cudaMemcpy(d_firstCellParticle + pNumCells,
                          d_firstCellParticle + pNumCells - 1,
                          sizeof(int), cudaMemcpyDeviceToDevice));

    k_fillParticles<<<pgrid(numParticles), BS>>>(d_posX, d_posY,
        d_firstCellParticle, d_cellParticleIds, numParticles);

    float *pinX = d_posX, *pinY = d_posY, *poutX = d_posX2, *poutY = d_posY2;
    float *cinR = d_colR, *cinG = d_colG, *cinB = d_colB;
    float *coutR = d_colR2, *coutG = d_colG2, *coutB = d_colB2;
    for (int it = 0; it < numIters; ++it) {
        k_pushIteration<<<pgrid(numParticles), BS>>>(pinX, pinY, cinR, cinG, cinB,
            poutX, poutY, coutR, coutG, coutB,
            d_firstCellParticle, d_cellParticleIds, numParticles);
        std::swap(pinX, poutX); std::swap(pinY, poutY);
        std::swap(cinR, coutR); std::swap(cinG, coutG); std::swap(cinB, coutB);
    }
    // Latest result lives in the "in" pointers after the final swap.
    if (pinX != d_posX) {
        size_t np = (size_t)numParticles * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_posX, pinX, np, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_posY, pinY, np, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_colR, cinR, np, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_colG, cinG, np, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_colB, cinB, np, cudaMemcpyDeviceToDevice));
    }
}

void FlipFluidCuda::transferToGrid() {
    int gc = pgrid(fNumCells), gp_ = pgrid(numParticles);
    k_savePrev<<<gc, BS>>>(d_u, d_v, d_du, d_dv, d_prevU, d_prevV, fNumCells);
    k_classifyInit<<<gc, BS>>>(d_s, d_cellType, fNumCells);
    k_classifyScatter<<<gp_, BS>>>(d_posX, d_posY, d_cellType, numParticles);
    k_p2g<<<gp_, BS>>>(0, d_posX, d_posY, d_velX, d_u, d_du, numParticles);
    k_p2g<<<gp_, BS>>>(1, d_posX, d_posY, d_velY, d_v, d_dv, numParticles);
    k_p2gNormalize<<<gc, BS>>>(d_u, d_du, fNumCells);
    k_p2gNormalize<<<gc, BS>>>(d_v, d_dv, fNumCells);
    k_restoreSolid<<<gc, BS>>>(d_u, d_v, d_prevU, d_prevV, d_cellType, fNumCells);
}

void FlipFluidCuda::densityStep() {
    CUDA_CHECK(cudaMemset(d_particleDensity, 0, (size_t)fNumCells * sizeof(float)));
    k_particleDensity<<<pgrid(numParticles), BS>>>(d_posX, d_posY, d_particleDensity, numParticles);
    if (particleRestDensity == 0.0f) {
        CUDA_CHECK(cudaDeviceSynchronize());
        particleRestDensity = computeRestDensityGPU();
    }
}

void FlipFluidCuda::solvePressure(int numIters, float dt, float overRelaxation,
                                  bool compensateDrift) {
    int gc = pgrid(fNumCells);
    float cp = density * h / dt;
    int cd = compensateDrift ? 1 : 0;
    k_solveInit<<<gc, BS>>>(d_u, d_v, d_p, d_prevU, d_prevV, fNumCells);

    if (solver == SOLVER_RED_BLACK) {
        for (int it = 0; it < numIters; ++it) {
            k_solveRB<<<gc, BS>>>(0, d_u, d_v, d_p, d_s, d_cellType,
                d_particleDensity, overRelaxation, cp, particleRestDensity, cd);
            k_solveRB<<<gc, BS>>>(1, d_u, d_v, d_p, d_s, d_cellType,
                d_particleDensity, overRelaxation, cp, particleRestDensity, cd);
        }
    } else {
        for (int it = 0; it < numIters; ++it) {
            CUDA_CHECK(cudaMemset(d_du, 0, (size_t)fNumCells * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_dv, 0, (size_t)fNumCells * sizeof(float)));
            k_jacobiAccum<<<gc, BS>>>(d_u, d_v, d_p, d_du, d_dv, d_s, d_cellType,
                d_particleDensity, overRelaxation, cp, particleRestDensity, cd);
            k_jacobiApply<<<gc, BS>>>(d_u, d_v, d_du, d_dv, fNumCells);
        }
    }
}

void FlipFluidCuda::transferFromGrid(float flipRatio) {
    int gp_ = pgrid(numParticles);
    k_g2p<<<gp_, BS>>>(0, d_posX, d_posY, d_velX, d_u, d_prevU, d_cellType,
                       flipRatio, numParticles, fNumCells);
    k_g2p<<<gp_, BS>>>(1, d_posX, d_posY, d_velY, d_v, d_prevV, d_cellType,
                       flipRatio, numParticles, fNumCells);
}

void FlipFluidCuda::colorStep() {
    k_updateParticleColors<<<pgrid(numParticles), BS>>>(d_posX, d_posY,
        d_colR, d_colG, d_colB, d_particleDensity, particleRestDensity, numParticles);
    k_updateCellColors<<<pgrid(fNumCells), BS>>>(d_cellType, d_particleDensity,
        d_cellColor, particleRestDensity, fNumCells);
}

void FlipFluidCuda::simulate(float dt, float gravity, float flipRatio,
                             int numPressureIters, int numParticleIters,
                             float overRelaxation, bool compensateDrift,
                             bool separateParticles,
                             float obstacleX, float obstacleY, float obstacleRadius,
                             float obstacleVelX, float obstacleVelY, int numSubSteps) {
    if (numSubSteps < 1) numSubSteps = 1;
    float sdt = dt / numSubSteps;
    if (timers) timers->numPressureIters = numPressureIters;

    for (int step = 0; step < numSubSteps; ++step) {
        CUTIME(T1_integrate,
            k_integrate<<<pgrid(numParticles), BS>>>(d_posX, d_posY, d_velX, d_velY,
                                                     numParticles, sdt, gravity));
        if (separateParticles)
            CUTIME(T2_pushApart, pushParticlesApart(numParticleIters));
        CUTIME(T3_collisions,
            k_collide<<<pgrid(numParticles), BS>>>(d_posX, d_posY, d_velX, d_velY,
                numParticles, obstacleX, obstacleY, obstacleRadius,
                obstacleVelX, obstacleVelY));
        CUTIME(T4_p2g,      transferToGrid());
        CUTIME(T5_density,  densityStep());
        CUTIME(T6_pressure, solvePressure(numPressureIters, sdt, overRelaxation,
                                          compensateDrift));
        CUTIME(T7_g2p,      transferFromGrid(flipRatio));
    }
    CUTIME(T8_colors, colorStep());

    // In untimed (interactive) mode the launches above are async; make the
    // frame's results available before the caller reads them back.
    if (!timers) cudaDeviceSynchronize();
}

void FlipFluidCuda::downloadForRender(flipcpu::FlipFluid& host) {
    CUTIME(T10_d2h, {
        size_t np = (size_t)numParticles * sizeof(float);
        cudaMemcpy(host.particlePosX.data(), d_posX, np, cudaMemcpyDeviceToHost);
        cudaMemcpy(host.particlePosY.data(), d_posY, np, cudaMemcpyDeviceToHost);
        cudaMemcpy(host.particleColorR.data(), d_colR, np, cudaMemcpyDeviceToHost);
        cudaMemcpy(host.particleColorG.data(), d_colG, np, cudaMemcpyDeviceToHost);
        cudaMemcpy(host.particleColorB.data(), d_colB, np, cudaMemcpyDeviceToHost);
        cudaMemcpy(host.cellColor.data(), d_cellColor,
                   (size_t)3 * fNumCells * sizeof(float), cudaMemcpyDeviceToHost);
    });
}

void FlipFluidCuda::downloadParticles(flipcpu::FlipFluid& host) {
    size_t np = (size_t)numParticles * sizeof(float);
    cudaMemcpy(host.particlePosX.data(), d_posX, np, cudaMemcpyDeviceToHost);
    cudaMemcpy(host.particlePosY.data(), d_posY, np, cudaMemcpyDeviceToHost);
    cudaMemcpy(host.particleVelX.data(), d_velX, np, cudaMemcpyDeviceToHost);
    cudaMemcpy(host.particleVelY.data(), d_velY, np, cudaMemcpyDeviceToHost);
}

} // namespace flipgpu
