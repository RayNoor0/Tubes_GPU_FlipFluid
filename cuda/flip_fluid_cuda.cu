// CUDA implementation of the FLIP/PIC solver. Each kernel is a direct translation
// of the corresponding flipcpu::FlipFluid method; see flip_fluid.cpp for the
// reference. Deviations forced by parallelism (red-black pressure, Jacobi
// separation) are noted at the relevant kernels.

#include "flip_fluid_cuda.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace flipcuda {

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float clampfd(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}
__device__ __forceinline__ int clampid(int x, int lo, int hi) {
    return max(lo, min(hi, x));
}

// --------------------------- T1: integrate ----------------------------------
__global__ void integrateKernel(float2* pos, float2* vel, int n,
                                 float dt, float gravity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vel[i].y += dt * gravity;
    pos[i].x += vel[i].x * dt;
    pos[i].y += vel[i].y * dt;
}

// --------------------------- T3: collisions ----------------------------------
__global__ void collisionsKernel(float2* pos, float2* vel, int n,
                                  float fInvSpacing, int fNumX, int fNumY,
                                  float particleRadius,
                                  float ox, float oy, float orad,
                                  float ovx, float ovy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float hh = 1.0f / fInvSpacing;
    float r = particleRadius;
    float minDist = orad + r;
    float minDist2 = minDist * minDist;
    float minX = hh + r, maxX = (fNumX - 1) * hh - r;
    float minY = hh + r, maxY = (fNumY - 1) * hh - r;

    float x = pos[i].x, y = pos[i].y;
    float dx = x - ox, dy = y - oy;
    if (dx * dx + dy * dy < minDist2) { vel[i].x = ovx; vel[i].y = ovy; }
    if (x < minX) { x = minX; vel[i].x = 0.0f; }
    if (x > maxX) { x = maxX; vel[i].x = 0.0f; }
    if (y < minY) { y = minY; vel[i].y = 0.0f; }
    if (y > maxY) { y = maxY; vel[i].y = 0.0f; }
    pos[i].x = x; pos[i].y = y;
}

// --------------------------- T4: P2G group -----------------------------------
__global__ void savePrevKernel(float* u, float* v, float* du, float* dv,
                               float* prevU, float* prevV, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    prevU[i] = u[i]; prevV[i] = v[i];
    du[i] = 0.0f; dv[i] = 0.0f; u[i] = 0.0f; v[i] = 0.0f;
}

__global__ void classifyGridKernel(int* cellType, const float* s, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    cellType[i] = (s[i] == 0.0f) ? SOLID_CELL : AIR_CELL;
}

// Idempotent AIR->FLUID write; many particles may hit the same cell but all write
// the same value, so no atomic is needed.
__global__ void classifyParticlesKernel(int* cellType, const float2* pos, int np,
                                         float fInvSpacing, int fNumX, int fNumY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    int xi = clampid((int)floorf(pos[i].x * fInvSpacing), 0, fNumX - 1);
    int yi = clampid((int)floorf(pos[i].y * fInvSpacing), 0, fNumY - 1);
    int c = xi * fNumY + yi;
    if (cellType[c] == AIR_CELL) cellType[c] = FLUID_CELL;
}

__global__ void p2gKernel(int component, const float2* pos, const float2* vel,
                          float* fld, float* fldD, int np,
                          int fNumX, int fNumY, float h, float fInvSpacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    int n = fNumY;
    float hh = h, h1 = fInvSpacing, h2 = 0.5f * h;
    float dxOff = (component == 0) ? 0.0f : h2;
    float dyOff = (component == 0) ? h2   : 0.0f;

    float x = clampfd(pos[i].x, hh, (fNumX - 1) * hh);
    float y = clampfd(pos[i].y, hh, (fNumY - 1) * hh);

    int x0 = min((int)floorf((x - dxOff) * h1), fNumX - 2);
    float tx = ((x - dxOff) - x0 * hh) * h1;
    int x1 = min(x0 + 1, fNumX - 2);
    int y0 = min((int)floorf((y - dyOff) * h1), fNumY - 2);
    float ty = ((y - dyOff) - y0 * hh) * h1;
    int y1 = min(y0 + 1, fNumY - 2);

    float sx = 1.0f - tx, sy = 1.0f - ty;
    float d0 = sx * sy, d1 = tx * sy, d2 = tx * ty, d3 = sx * ty;
    int nr0 = x0 * n + y0, nr1 = x1 * n + y0, nr2 = x1 * n + y1, nr3 = x0 * n + y1;

    float pv = (component == 0) ? vel[i].x : vel[i].y;
    atomicAdd(&fld[nr0], pv * d0); atomicAdd(&fldD[nr0], d0);
    atomicAdd(&fld[nr1], pv * d1); atomicAdd(&fldD[nr1], d1);
    atomicAdd(&fld[nr2], pv * d2); atomicAdd(&fldD[nr2], d2);
    atomicAdd(&fld[nr3], pv * d3); atomicAdd(&fldD[nr3], d3);
}

__global__ void normalizeKernel(float* fld, const float* fldD, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    if (fldD[i] > 0.0f) fld[i] /= fldD[i];
}

__global__ void restoreSolidKernel(float* u, float* v,
                                   const float* prevU, const float* prevV,
                                   const int* cellType, int fNumX, int fNumY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fNumX * fNumY) return;
    int n = fNumY, i = idx / n, j = idx % n;
    bool solid = (cellType[i * n + j] == SOLID_CELL);
    if (solid || (i > 0 && cellType[(i - 1) * n + j] == SOLID_CELL))
        u[i * n + j] = prevU[i * n + j];
    if (solid || (j > 0 && cellType[i * n + j - 1] == SOLID_CELL))
        v[i * n + j] = prevV[i * n + j];
}

// --------------------------- T5: density -------------------------------------
__global__ void densityKernel(const float2* pos, float* particleDensity, int np,
                              int fNumX, int fNumY, float h, float fInvSpacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    int n = fNumY;
    float hh = h, h1 = fInvSpacing, h2 = 0.5f * h;
    float x = clampfd(pos[i].x, hh, (fNumX - 1) * hh);
    float y = clampfd(pos[i].y, hh, (fNumY - 1) * hh);

    int x0 = (int)floorf((x - h2) * h1);
    float tx = ((x - h2) - x0 * hh) * h1;
    int x1 = min(x0 + 1, fNumX - 2);
    int y0 = (int)floorf((y - h2) * h1);
    float ty = ((y - h2) - y0 * hh) * h1;
    int y1 = min(y0 + 1, fNumY - 2);

    float sx = 1.0f - tx, sy = 1.0f - ty;
    if (x0 < fNumX && y0 < fNumY) atomicAdd(&particleDensity[x0 * n + y0], sx * sy);
    if (x1 < fNumX && y0 < fNumY) atomicAdd(&particleDensity[x1 * n + y0], tx * sy);
    if (x1 < fNumX && y1 < fNumY) atomicAdd(&particleDensity[x1 * n + y1], tx * ty);
    if (x0 < fNumX && y1 < fNumY) atomicAdd(&particleDensity[x0 * n + y1], sx * ty);
}

__global__ void restDensityKernel(const float* particleDensity, const int* cellType,
                                  int nc, float* sum, int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    if (cellType[i] == FLUID_CELL) {
        atomicAdd(sum, particleDensity[i]);
        atomicAdd(count, 1);
    }
}

// --------------------------- T6: pressure (red-black GS) ---------------------
__global__ void pressureInitKernel(float* p, const float* u, const float* v,
                                   float* prevU, float* prevV, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    p[i] = 0.0f; prevU[i] = u[i]; prevV[i] = v[i];
}

// Red-black Gauss-Seidel: one launch updates only fluid cells whose (i+j) parity
// matches `color`, in place, exactly like the CPU sweep (same overRelaxation + drift
// compensation). On a MAC grid each velocity face is incident to exactly one red and
// one black cell, so within a single color pass every face has at most one writer ->
// race-free WITHOUT atomics, and deterministic. Two passes per iteration replace the
// CPU's lexicographic sweep; same omega (1.9) and iteration count keep parity close.
__global__ void pressureRBKernel(int color, float* u, float* v, float* p,
                                 const float* s, const int* cellType,
                                 const float* particleDensity,
                                 int fNumX, int fNumY,
                                 float cp, float rest, int cd, float overRelaxation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fNumX * fNumY) return;
    int n = fNumY, i = idx / n, j = idx % n;
    if (i < 1 || i >= fNumX - 1 || j < 1 || j >= fNumY - 1) return;
    if (((i + j) & 1) != color) return;

    int center = i * n + j;
    if (cellType[center] != FLUID_CELL) return;
    int left = (i - 1) * n + j, right = (i + 1) * n + j;
    int bottom = i * n + j - 1, top = i * n + j + 1;

    float sx0 = s[left], sx1 = s[right], sy0 = s[bottom], sy1 = s[top];
    float sSum = sx0 + sx1 + sy0 + sy1;
    if (sSum == 0.0f) return;

    float div = u[right] - u[center] + v[top] - v[center];
    if (rest > 0.0f && cd != 0) {
        float compression = particleDensity[center] - rest;
        if (compression > 0.0f) div = div - compression; // k = 1.0
    }
    float pVal = -div / sSum;
    pVal *= overRelaxation;
    p[center] += cp * pVal;
    u[center] -= sx0 * pVal;
    u[right]  += sx1 * pVal;
    v[center] -= sy0 * pVal;
    v[top]    += sy1 * pVal;
}

// --------------------------- T7: G2P -----------------------------------------
__global__ void g2pKernel(int component, const float2* pos, float2* vel,
                          const float* fld, const float* pfld, const int* cellType,
                          int np, int fNumX, int fNumY, float h, float fInvSpacing,
                          float flipRatio) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    int n = fNumY;
    float hh = h, h1 = fInvSpacing, h2 = 0.5f * h;
    float dxOff = (component == 0) ? 0.0f : h2;
    float dyOff = (component == 0) ? h2   : 0.0f;
    int offset = (component == 0) ? n : 1;

    float x = clampfd(pos[i].x, hh, (fNumX - 1) * hh);
    float y = clampfd(pos[i].y, hh, (fNumY - 1) * hh);

    int x0 = min((int)floorf((x - dxOff) * h1), fNumX - 2);
    float tx = ((x - dxOff) - x0 * hh) * h1;
    int x1 = min(x0 + 1, fNumX - 2);
    int y0 = min((int)floorf((y - dyOff) * h1), fNumY - 2);
    float ty = ((y - dyOff) - y0 * hh) * h1;
    int y1 = min(y0 + 1, fNumY - 2);

    float sx = 1.0f - tx, sy = 1.0f - ty;
    float d0 = sx * sy, d1 = tx * sy, d2 = tx * ty, d3 = sx * ty;
    int nr0 = x0 * n + y0, nr1 = x1 * n + y0, nr2 = x1 * n + y1, nr3 = x0 * n + y1;

    float valid0 = (cellType[nr0] != AIR_CELL || cellType[nr0 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid1 = (cellType[nr1] != AIR_CELL || cellType[nr1 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid2 = (cellType[nr2] != AIR_CELL || cellType[nr2 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid3 = (cellType[nr3] != AIR_CELL || cellType[nr3 - offset] != AIR_CELL) ? 1.0f : 0.0f;

    float v_old = (component == 0) ? vel[i].x : vel[i].y;
    float d = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;
    if (d > 0.0f) {
        float f0 = fld[nr0], f1 = fld[nr1], f2 = fld[nr2], f3 = fld[nr3];
        float pf0 = pfld[nr0], pf1 = pfld[nr1], pf2 = pfld[nr2], pf3 = pfld[nr3];
        float picV = (valid0 * d0 * f0 + valid1 * d1 * f1
                    + valid2 * d2 * f2 + valid3 * d3 * f3) / d;
        float corr = (valid0 * d0 * (f0 - pf0) + valid1 * d1 * (f1 - pf1)
                    + valid2 * d2 * (f2 - pf2) + valid3 * d3 * (f3 - pf3)) / d;
        float flipV = v_old + corr;
        float blended = (1.0f - flipRatio) * picV + flipRatio * flipV;
        if (component == 0) vel[i].x = blended; else vel[i].y = blended;
    }
}

// --------------------------- T8: colors --------------------------------------
__global__ void particleColorsKernel(const float2* pos, float3* col,
                                      const float* particleDensity, int np,
                                      int fNumX, int fNumY, float fInvSpacing,
                                      float restDensity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    const float sStep = 0.01f;
    float r = clampfd(col[i].x - sStep, 0.0f, 1.0f);
    float g = clampfd(col[i].y - sStep, 0.0f, 1.0f);
    float b = clampfd(col[i].z + sStep, 0.0f, 1.0f);

    int xi = clampid((int)floorf(pos[i].x * fInvSpacing), 1, fNumX - 1);
    int yi = clampid((int)floorf(pos[i].y * fInvSpacing), 1, fNumY - 1);
    int c = xi * fNumY + yi;
    if (restDensity > 0.0f) {
        float rel = particleDensity[c] / restDensity;
        if (rel < 0.7f) { float s2 = 0.8f; r = s2; g = s2; b = 1.0f; }
    }
    col[i] = make_float3(r, g, b);
}

__device__ void setSciColorDev(float* cellColor, int cellNr, float val,
                               float minVal, float maxVal) {
    val = fminf(fmaxf(val, minVal), maxVal - 0.0001f);
    float d = maxVal - minVal;
    if (d == 0.0f) val = 0.5f; else val = (val - minVal) / d;
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
    cellColor[3 * cellNr] = r; cellColor[3 * cellNr + 1] = g; cellColor[3 * cellNr + 2] = b;
}

__global__ void cellColorsKernel(float* cellColor, const int* cellType,
                                 const float* particleDensity, int nc,
                                 float restDensity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    if (cellType[i] == SOLID_CELL) {
        cellColor[3 * i] = 0.5f; cellColor[3 * i + 1] = 0.5f; cellColor[3 * i + 2] = 0.5f;
    } else if (cellType[i] == FLUID_CELL) {
        float d = particleDensity[i];
        if (restDensity > 0.0f) d /= restDensity;
        setSciColorDev(cellColor, i, d, 0.0f, 2.0f);
    } else {
        cellColor[3 * i] = 0.0f; cellColor[3 * i + 1] = 0.0f; cellColor[3 * i + 2] = 0.0f;
    }
}

// --------------------------- T2: push apart ----------------------------------
__global__ void zeroIntKernel(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0;
}

__global__ void countKernel(const float2* pos, int* counts, int np,
                            float pInvSpacing, int pNumX, int pNumY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    int xi = clampid((int)floorf(pos[i].x * pInvSpacing), 0, pNumX - 1);
    int yi = clampid((int)floorf(pos[i].y * pInvSpacing), 0, pNumY - 1);
    atomicAdd(&counts[xi * pNumY + yi], 1);
}

__global__ void setLastKernel(int* first, const int* counts, int pNumCells) {
    first[pNumCells] = first[pNumCells - 1] + counts[pNumCells - 1];
}

__global__ void scatterKernel(const float2* pos, int* cursor, int* ids, int np,
                              float pInvSpacing, int pNumX, int pNumY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    int xi = clampid((int)floorf(pos[i].x * pInvSpacing), 0, pNumX - 1);
    int yi = clampid((int)floorf(pos[i].y * pInvSpacing), 0, pNumY - 1);
    int c = xi * pNumY + yi;
    int idx = atomicAdd(&cursor[c], 1);
    ids[idx] = i;
}

// Jacobi separation: every particle reads neighbour positions/colors from the
// fixed snapshot (posIn/colIn) and writes only its OWN corrected value. This
// replaces the CPU's in-place sequential push (which also moved the neighbour);
// symmetric correction is recovered because each neighbour applies its own.
__global__ void separateKernel(const float2* posIn, const float3* colIn,
                               float2* posOut, float3* colOut,
                               const int* firstCellParticle, const int* ids, int np,
                               float pInvSpacing, int pNumX, int pNumY,
                               float particleRadius, float colorDiffusionCoeff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    float px = posIn[i].x, py = posIn[i].y;
    float3 ci = colIn[i];
    float minDist = 2.0f * particleRadius;
    float minDist2 = minDist * minDist;

    int pxi = (int)floorf(px * pInvSpacing);
    int pyi = (int)floorf(py * pInvSpacing);
    int x0 = max(pxi - 1, 0), y0 = max(pyi - 1, 0);
    int x1 = min(pxi + 1, pNumX - 1), y1 = min(pyi + 1, pNumY - 1);

    float ddx = 0.0f, ddy = 0.0f;
    float3 dc = make_float3(0.0f, 0.0f, 0.0f);
    for (int xi = x0; xi <= x1; ++xi) {
        for (int yi = y0; yi <= y1; ++yi) {
            int c = xi * pNumY + yi;
            int first = firstCellParticle[c];
            int last  = firstCellParticle[c + 1];
            for (int j = first; j < last; ++j) {
                int idn = ids[j];
                if (idn == i) continue;
                float qx = posIn[idn].x, qy = posIn[idn].y;
                float dx = qx - px, dy = qy - py;
                float d2 = dx * dx + dy * dy;
                if (d2 > minDist2 || d2 == 0.0f) continue;
                float d = sqrtf(d2);
                float sFac = 0.5f * (minDist - d) / d;
                ddx -= dx * sFac;  // CPU: particlePosX[i] -= (qx-px)*sFac
                ddy -= dy * sFac;
                float3 cj = colIn[idn];
                float cr = (ci.x + cj.x) * 0.5f;
                float cg = (ci.y + cj.y) * 0.5f;
                float cb = (ci.z + cj.z) * 0.5f;
                dc.x += (cr - ci.x) * colorDiffusionCoeff;
                dc.y += (cg - ci.y) * colorDiffusionCoeff;
                dc.z += (cb - ci.z) * colorDiffusionCoeff;
            }
        }
    }
    posOut[i] = make_float2(px + ddx, py + ddy);
    colOut[i] = make_float3(ci.x + dc.x, ci.y + dc.y, ci.z + dc.z);
}

// --------------------------- obstacle carve ----------------------------------
__global__ void carveKernel(float* s, float* u, float* v, int fNumX, int fNumY,
                            float h, float x, float y, float r, float vx, float vy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fNumX * fNumY) return;
    int n = fNumY, i = idx / n, j = idx % n;
    if (i < 1 || i >= fNumX - 2 || j < 1 || j >= fNumY - 2) return; // matches CPU bounds
    s[i * n + j] = 1.0f;
    float dx = (i + 0.5f) * h - x;
    float dy = (j + 0.5f) * h - y;
    if (dx * dx + dy * dy < r * r) {
        s[i * n + j] = 0.0f;
        u[i * n + j] = vx; u[(i + 1) * n + j] = vx;
        v[i * n + j] = vy; v[i * n + j + 1] = vy;
    }
}

// ============================ class methods ==================================

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

    auto allocF = [&](float** p, int n) {
        CUDA_CHECK(cudaMalloc(p, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(*p, 0, n * sizeof(float)));
    };
    auto allocI = [&](int** p, int n) {
        CUDA_CHECK(cudaMalloc(p, n * sizeof(int)));
        CUDA_CHECK(cudaMemset(*p, 0, n * sizeof(int)));
    };

    allocF(&d_u, fNumCells);  allocF(&d_v, fNumCells);
    allocF(&d_du, fNumCells); allocF(&d_dv, fNumCells);
    allocF(&d_prevU, fNumCells); allocF(&d_prevV, fNumCells);
    allocF(&d_p, fNumCells);  allocF(&d_s, fNumCells);
    allocI(&d_cellType, fNumCells);
    allocF(&d_particleDensity, fNumCells);
    allocF(&d_cellColor, 3 * fNumCells);

    CUDA_CHECK(cudaMalloc(&d_vel, maxParticles * sizeof(float2)));
    CUDA_CHECK(cudaMemset(d_vel, 0, maxParticles * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_posScratch, maxParticles * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_colScratch, maxParticles * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_pos, maxParticles * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_col, maxParticles * sizeof(float3)));
    hostPos.assign(2 * maxParticles, 0.0f);
    hostCol.assign(3 * maxParticles, 0.0f);

    allocI(&d_numCellParticles, pNumCells);
    allocI(&d_firstCellParticle, pNumCells + 1);
    allocI(&d_cellParticleIds, maxParticles);
    allocI(&d_cellCursor, pNumCells);

    allocF(&d_sum, 1);
    allocI(&d_count, 1);

    for (int s = 0; s < fliptiming::STAGE_COUNT; ++s) {
        CUDA_CHECK(cudaEventCreate(&evStart[s]));
        CUDA_CHECK(cudaEventCreate(&evStop[s]));
    }
}

FlipFluidCuda::~FlipFluidCuda() {
    if (posRes) cudaGraphicsUnregisterResource(posRes);
    if (colRes) cudaGraphicsUnregisterResource(colRes);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_du); cudaFree(d_dv);
    cudaFree(d_prevU); cudaFree(d_prevV); cudaFree(d_p); cudaFree(d_s);
    cudaFree(d_cellType); cudaFree(d_particleDensity); cudaFree(d_cellColor);
    cudaFree(d_vel); cudaFree(d_posScratch); cudaFree(d_colScratch);
    cudaFree(d_pos); cudaFree(d_col);
    cudaFree(d_numCellParticles); cudaFree(d_firstCellParticle);
    cudaFree(d_cellParticleIds); cudaFree(d_cellCursor);
    cudaFree(d_sum); cudaFree(d_count);
    for (int s = 0; s < fliptiming::STAGE_COUNT; ++s) {
        cudaEventDestroy(evStart[s]); cudaEventDestroy(evStop[s]);
    }
}

bool FlipFluidCuda::tryRegisterVBOs(GLuint pos, GLuint col) {
    posVBO = pos; colVBO = col;
    cudaError_t e1 = cudaGraphicsGLRegisterBuffer(&posRes, posVBO,
                                                  cudaGraphicsRegisterFlagsNone);
    if (e1 != cudaSuccess) {
        posRes = nullptr;
        cudaGetLastError();           // clear pending error so later CUDA_CHECKs are clean
        interopEnabled = false;
        return false;
    }
    cudaError_t e2 = cudaGraphicsGLRegisterBuffer(&colRes, colVBO,
                                                  cudaGraphicsRegisterFlagsNone);
    if (e2 != cudaSuccess) {
        cudaGraphicsUnregisterResource(posRes);
        posRes = nullptr; colRes = nullptr;
        cudaGetLastError();
        interopEnabled = false;
        return false;
    }
    interopEnabled = true;
    return true;
}

void FlipFluidCuda::uploadParticles(const std::vector<float>& posXY,
                                    const std::vector<float>& colRGB) {
    // Keep a host copy (used for the no-interop render path and the frame-0 seed)...
    hostPos = posXY;
    hostCol = colRGB;
    hostPos.resize(2 * maxParticles, 0.0f);
    hostCol.resize(3 * maxParticles, 0.0f);
    // ...and seed the owned device buffers.
    CUDA_CHECK(cudaMemcpy(d_pos, hostPos.data(), maxParticles * sizeof(float2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, hostCol.data(), maxParticles * sizeof(float3),
                          cudaMemcpyHostToDevice));
}

void FlipFluidCuda::uploadSolid(const std::vector<float>& s) {
    CUDA_CHECK(cudaMemcpy(d_s, s.data(), fNumCells * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void FlipFluidCuda::resetVelocities() {
    CUDA_CHECK(cudaMemset(d_vel, 0, maxParticles * sizeof(float2)));
}

void FlipFluidCuda::carveObstacle(float x, float y, float r, float vx, float vy) {
    int blocks = divUp(fNumCells, BLOCK);
    carveKernel<<<blocks, BLOCK>>>(d_s, d_u, d_v, fNumX, fNumY, h, x, y, r, vx, vy);
}

void FlipFluidCuda::mapVBOs(float2** outPos, float3** outCol) {
    CUDA_CHECK(cudaGraphicsMapResources(1, &posRes, 0));
    CUDA_CHECK(cudaGraphicsMapResources(1, &colRes, 0));
    size_t bytes = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)outPos, &bytes, posRes));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)outCol, &bytes, colRes));
}

void FlipFluidCuda::unmapVBOs() {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &posRes, 0));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &colRes, 0));
}

void FlipFluidCuda::k_integrate(float2* pos, float dt, float gravity) {
    integrateKernel<<<divUp(numParticles, BLOCK), BLOCK>>>(pos, d_vel, numParticles,
                                                           dt, gravity);
}

void FlipFluidCuda::k_collisions(float2* pos, float ox, float oy, float orad,
                                 float ovx, float ovy) {
    collisionsKernel<<<divUp(numParticles, BLOCK), BLOCK>>>(
        pos, d_vel, numParticles, fInvSpacing, fNumX, fNumY, particleRadius,
        ox, oy, orad, ovx, ovy);
}

void FlipFluidCuda::transferToGrid(float2* pos) {
    int gc = divUp(fNumCells, BLOCK);
    int pc = divUp(numParticles, BLOCK);
    savePrevKernel<<<gc, BLOCK>>>(d_u, d_v, d_du, d_dv, d_prevU, d_prevV, fNumCells);
    classifyGridKernel<<<gc, BLOCK>>>(d_cellType, d_s, fNumCells);
    classifyParticlesKernel<<<pc, BLOCK>>>(d_cellType, pos, numParticles,
                                           fInvSpacing, fNumX, fNumY);
    p2gKernel<<<pc, BLOCK>>>(U_FIELD, pos, d_vel, d_u, d_du, numParticles,
                             fNumX, fNumY, h, fInvSpacing);
    p2gKernel<<<pc, BLOCK>>>(V_FIELD, pos, d_vel, d_v, d_dv, numParticles,
                             fNumX, fNumY, h, fInvSpacing);
    normalizeKernel<<<gc, BLOCK>>>(d_u, d_du, fNumCells);
    normalizeKernel<<<gc, BLOCK>>>(d_v, d_dv, fNumCells);
    restoreSolidKernel<<<gc, BLOCK>>>(d_u, d_v, d_prevU, d_prevV, d_cellType,
                                      fNumX, fNumY);
}

void FlipFluidCuda::updateDensity(float2* pos) {
    CUDA_CHECK(cudaMemset(d_particleDensity, 0, fNumCells * sizeof(float)));
    densityKernel<<<divUp(numParticles, BLOCK), BLOCK>>>(
        pos, d_particleDensity, numParticles, fNumX, fNumY, h, fInvSpacing);
}

void FlipFluidCuda::computeRestDensityIfNeeded() {
    if (restDensityComputed) return;
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    restDensityKernel<<<divUp(fNumCells, BLOCK), BLOCK>>>(d_particleDensity,
                                                          d_cellType, fNumCells,
                                                          d_sum, d_count);
    float sum = 0.0f; int count = 0;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    particleRestDensity = (count > 0) ? (sum / count) : 0.0f;
    restDensityComputed = true;
}

void FlipFluidCuda::solvePressure(int numIters, float dt, float overRelaxation,
                                  bool compensateDrift) {
    int gc = divUp(fNumCells, BLOCK);
    float cp = density * h / dt;
    int cd = compensateDrift ? 1 : 0;

    // pressureInit sets p=0 and snapshots the post-P2G velocities into prevU/prevV
    // (used later by g2p for the FLIP correction).
    pressureInitKernel<<<gc, BLOCK>>>(d_p, d_u, d_v, d_prevU, d_prevV, fNumCells);

    // Red-black Gauss-Seidel: red pass then black pass, in place, per iteration.
    for (int iter = 0; iter < numIters; ++iter) {
        pressureRBKernel<<<gc, BLOCK>>>(0, d_u, d_v, d_p, d_s, d_cellType,
                                        d_particleDensity, fNumX, fNumY,
                                        cp, particleRestDensity, cd, overRelaxation);
        pressureRBKernel<<<gc, BLOCK>>>(1, d_u, d_v, d_p, d_s, d_cellType,
                                        d_particleDensity, fNumX, fNumY,
                                        cp, particleRestDensity, cd, overRelaxation);
    }
}

void FlipFluidCuda::transferToParticles(float2* pos, float flipRatio) {
    int pc = divUp(numParticles, BLOCK);
    g2pKernel<<<pc, BLOCK>>>(U_FIELD, pos, d_vel, d_u, d_prevU, d_cellType,
                             numParticles, fNumX, fNumY, h, fInvSpacing, flipRatio);
    g2pKernel<<<pc, BLOCK>>>(V_FIELD, pos, d_vel, d_v, d_prevV, d_cellType,
                             numParticles, fNumX, fNumY, h, fInvSpacing, flipRatio);
}

void FlipFluidCuda::updateColors(float2* pos, float3* col) {
    particleColorsKernel<<<divUp(numParticles, BLOCK), BLOCK>>>(
        pos, col, d_particleDensity, numParticles, fNumX, fNumY, fInvSpacing,
        particleRestDensity);
    cellColorsKernel<<<divUp(fNumCells, BLOCK), BLOCK>>>(
        d_cellColor, d_cellType, d_particleDensity, fNumCells, particleRestDensity);
}

void FlipFluidCuda::pushApart(float2* pos, float3* col, int numIters) {
    int pc = divUp(numParticles, BLOCK);
    // Build the spatial hash once (matches CPU: bins fixed across iterations).
    zeroIntKernel<<<divUp(pNumCells, BLOCK), BLOCK>>>(d_numCellParticles, pNumCells);
    countKernel<<<pc, BLOCK>>>(pos, d_numCellParticles, numParticles,
                               pInvSpacing, pNumX, pNumY);
    thrust::device_ptr<int> cnt(d_numCellParticles);
    thrust::device_ptr<int> out(d_firstCellParticle);
    thrust::exclusive_scan(thrust::device, cnt, cnt + pNumCells, out);
    setLastKernel<<<1, 1>>>(d_firstCellParticle, d_numCellParticles, pNumCells);
    CUDA_CHECK(cudaMemcpy(d_cellCursor, d_firstCellParticle, pNumCells * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    scatterKernel<<<pc, BLOCK>>>(pos, d_cellCursor, d_cellParticleIds, numParticles,
                                 pInvSpacing, pNumX, pNumY);

    const float colorDiffusionCoeff = 0.001f;
    for (int iter = 0; iter < numIters; ++iter) {
        CUDA_CHECK(cudaMemcpy(d_posScratch, pos, numParticles * sizeof(float2),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_colScratch, col, numParticles * sizeof(float3),
                              cudaMemcpyDeviceToDevice));
        separateKernel<<<pc, BLOCK>>>(d_posScratch, d_colScratch, pos, col,
                                      d_firstCellParticle, d_cellParticleIds,
                                      numParticles, pInvSpacing, pNumX, pNumY,
                                      particleRadius, colorDiffusionCoeff);
    }
}

void FlipFluidCuda::updateCellColorsOnce() {
    cellColorsKernel<<<divUp(fNumCells, BLOCK), BLOCK>>>(
        d_cellColor, d_cellType, d_particleDensity, fNumCells, particleRestDensity);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void FlipFluidCuda::downloadParticles(std::vector<float>& posXY,
                                      std::vector<float>& velXY) {
    posXY.resize(2 * numParticles);
    velXY.resize(2 * numParticles);
    CUDA_CHECK(cudaMemcpy(velXY.data(), d_vel, numParticles * sizeof(float2),
                          cudaMemcpyDeviceToHost));
    if (interopEnabled) {
        float2* pos = nullptr; float3* col = nullptr;
        mapVBOs(&pos, &col);
        CUDA_CHECK(cudaMemcpy(posXY.data(), pos, numParticles * sizeof(float2),
                              cudaMemcpyDeviceToHost));
        unmapVBOs();
    } else {
        CUDA_CHECK(cudaMemcpy(posXY.data(), d_pos, numParticles * sizeof(float2),
                              cudaMemcpyDeviceToHost));
    }
}

void FlipFluidCuda::downloadCellColors(std::vector<float>& out) {
    out.resize(3 * fNumCells);
    CUDA_CHECK(cudaMemcpy(out.data(), d_cellColor, 3 * fNumCells * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void FlipFluidCuda::simulate(float dt, float gravity, float flipRatio,
                             int numPressureIters, int numParticleIters,
                             float overRelaxation, bool compensateDrift,
                             bool separateParticles,
                             float obstacleX, float obstacleY, float obstacleRadius,
                             float obstacleVelX, float obstacleVelY,
                             int numSubSteps) {
    using Clock = std::chrono::steady_clock;
    auto nowc = []{ return Clock::now(); };
    auto msc  = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Choose the render hand-off path. interop: kernels write the mapped GL VBOs.
    // no-interop (WSL2 fallback): kernels write the owned d_pos/d_col, copied D2H below.
    float2* pos = nullptr; float3* col = nullptr;
    double t10 = 0.0;
    if (interopEnabled) {
        auto m0 = nowc();
        mapVBOs(&pos, &col);          // T10 = map (+ unmap below)
        auto m1 = nowc();
        t10 = msc(m0, m1);
    } else {
        pos = d_pos; col = d_col;
    }

    if (numSubSteps < 1) numSubSteps = 1;
    float sdt = dt / numSubSteps;

    using fliptiming::Stage;
    auto recA = [&](Stage s){ cudaEventRecord(evStart[s]); };
    auto recB = [&](Stage s){ cudaEventRecord(evStop[s]); };

    for (int step = 0; step < numSubSteps; ++step) {
        recA(fliptiming::T1_integrate); k_integrate(pos, sdt, gravity); recB(fliptiming::T1_integrate);
        recA(fliptiming::T2_pushApart); if (separateParticles) pushApart(pos, col, numParticleIters); recB(fliptiming::T2_pushApart);
        recA(fliptiming::T3_collisions); k_collisions(pos, obstacleX, obstacleY, obstacleRadius, obstacleVelX, obstacleVelY); recB(fliptiming::T3_collisions);
        recA(fliptiming::T4_p2g); transferToGrid(pos); recB(fliptiming::T4_p2g);
        recA(fliptiming::T5_density); updateDensity(pos); computeRestDensityIfNeeded(); recB(fliptiming::T5_density);
        recA(fliptiming::T6_pressure); solvePressure(numPressureIters, sdt, overRelaxation, compensateDrift); recB(fliptiming::T6_pressure);
        recA(fliptiming::T7_g2p); transferToParticles(pos, flipRatio); recB(fliptiming::T7_g2p);

        // One sync per sub-step (all events on the default stream are then complete).
        CUDA_CHECK(cudaEventSynchronize(evStop[fliptiming::T7_g2p]));
        for (int s = fliptiming::T1_integrate; s <= fliptiming::T7_g2p; ++s) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart[s], evStop[s]);
            stageAdd(timing, (Stage)s, (double)ms);
        }
    }

    recA(fliptiming::T8_colors); updateColors(pos, col); recB(fliptiming::T8_colors);
    CUDA_CHECK(cudaEventSynchronize(evStop[fliptiming::T8_colors]));
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evStart[fliptiming::T8_colors], evStop[fliptiming::T8_colors]);
        stageAdd(timing, fliptiming::T8_colors, (double)ms);
    }

    // T10: interop -> unmap; no-interop -> device->host copy of the data we render.
    if (interopEnabled) {
        auto u0 = nowc();
        unmapVBOs();
        auto u1 = nowc();
        t10 += msc(u0, u1);
    } else {
        auto c0 = nowc();
        CUDA_CHECK(cudaMemcpy(hostPos.data(), d_pos, numParticles * sizeof(float2),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostCol.data(), d_col, numParticles * sizeof(float3),
                              cudaMemcpyDeviceToHost));
        auto c1 = nowc();
        t10 += msc(c0, c1);
    }
    stageAdd(timing, fliptiming::T10_transfer, t10);

    stageSetIters(timing, numPressureIters);
}

} // namespace flipcuda
