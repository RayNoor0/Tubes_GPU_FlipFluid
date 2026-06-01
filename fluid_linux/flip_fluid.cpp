

#include "flip_fluid.h"

#include <algorithm>
#include <cmath>

namespace flipcpu {

static inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}
static inline int clampi(int x, int lo, int hi) {
    return std::max(lo, std::min(hi, x));
}

FlipFluid::FlipFluid(float density_, float width, float height,
                     float spacing, float particle_radius, int max_particles)
{
    density = density_;
    fNumX = int(std::floor(width / spacing)) + 1;
    fNumY = int(std::floor(height / spacing)) + 1;
    h = std::max(width / fNumX, height / fNumY);
    fInvSpacing = 1.0f / h;
    fNumCells = fNumX * fNumY;

    u.assign(fNumCells, 0.0f);
    v.assign(fNumCells, 0.0f);
    du.assign(fNumCells, 0.0f);
    dv.assign(fNumCells, 0.0f);
    prevU.assign(fNumCells, 0.0f);
    prevV.assign(fNumCells, 0.0f);
    p.assign(fNumCells, 0.0f);
    s.assign(fNumCells, 0.0f);
    cellType.assign(fNumCells, AIR_CELL);
    cellColor.assign(3 * fNumCells, 0.0f);

    maxParticles = max_particles;
    particlePosX.assign(maxParticles, 0.0f);
    particlePosY.assign(maxParticles, 0.0f);
    particleVelX.assign(maxParticles, 0.0f);
    particleVelY.assign(maxParticles, 0.0f);
    particleColorR.assign(maxParticles, 0.0f);
    particleColorG.assign(maxParticles, 0.0f);
    particleColorB.assign(maxParticles, 1.0f);
    particleDensity.assign(fNumCells, 0.0f);
    particleRestDensity = 0.0f;

    particleRadius = particle_radius;
    pInvSpacing = 1.0f / (2.2f * particleRadius);
    pNumX = int(std::floor(width * pInvSpacing)) + 1;
    pNumY = int(std::floor(height * pInvSpacing)) + 1;
    pNumCells = pNumX * pNumY;

    numCellParticles.assign(pNumCells, 0);
    firstCellParticle.assign(pNumCells + 1, 0);
    cellParticleIds.assign(maxParticles, 0);

    numParticles = 0;
}

void FlipFluid::integrateParticles(float dt, float gravity) {
    for (int i = 0; i < numParticles; ++i) {
        particleVelY[i] += dt * gravity;
        particlePosX[i] += particleVelX[i] * dt;
        particlePosY[i] += particleVelY[i] * dt;
    }
}

void FlipFluid::pushParticlesApart(int numIters) {
    const float colorDiffusionCoeff = 0.001f;

    // count particles per cell
    std::fill(numCellParticles.begin(), numCellParticles.end(), 0);
    for (int i = 0; i < numParticles; ++i) {
        int xi = clampi(int(std::floor(particlePosX[i] * pInvSpacing)), 0, pNumX - 1);
        int yi = clampi(int(std::floor(particlePosY[i] * pInvSpacing)), 0, pNumY - 1);
        numCellParticles[xi * pNumY + yi] += 1;
    }

    // prefix sum (note: matches JS — firstCellParticle[i] holds sum thru i,
    // then we decrement during the fill pass).
    int first = 0;
    for (int i = 0; i < pNumCells; ++i) {
        first += numCellParticles[i];
        firstCellParticle[i] = first;
    }
    firstCellParticle[pNumCells] = first;

    // fill cell -> particles
    for (int i = 0; i < numParticles; ++i) {
        int xi = clampi(int(std::floor(particlePosX[i] * pInvSpacing)), 0, pNumX - 1);
        int yi = clampi(int(std::floor(particlePosY[i] * pInvSpacing)), 0, pNumY - 1);
        int cellNr = xi * pNumY + yi;
        firstCellParticle[cellNr] -= 1;
        cellParticleIds[firstCellParticle[cellNr]] = i;
    }

    // separate
    const float minDist = 2.0f * particleRadius;
    const float minDist2 = minDist * minDist;

    for (int iter = 0; iter < numIters; ++iter) {
        for (int i = 0; i < numParticles; ++i) {
            float px = particlePosX[i];
            float py = particlePosY[i];

            int pxi = int(std::floor(px * pInvSpacing));
            int pyi = int(std::floor(py * pInvSpacing));
            int x0 = std::max(pxi - 1, 0);
            int y0 = std::max(pyi - 1, 0);
            int x1 = std::min(pxi + 1, pNumX - 1);
            int y1 = std::min(pyi + 1, pNumY - 1);

            for (int xi = x0; xi <= x1; ++xi) {
                for (int yi = y0; yi <= y1; ++yi) {
                    int cellNr = xi * pNumY + yi;
                    int firstI = firstCellParticle[cellNr];
                    int lastI  = firstCellParticle[cellNr + 1];
                    for (int j = firstI; j < lastI; ++j) {
                        int idn = cellParticleIds[j];
                        if (idn == i) continue;
                        float qx = particlePosX[idn];
                        float qy = particlePosY[idn];

                        float dx = qx - px;
                        float dy = qy - py;
                        float d2 = dx * dx + dy * dy;
                        if (d2 > minDist2 || d2 == 0.0f) continue;
                        float d = std::sqrt(d2);
                        float sFac = 0.5f * (minDist - d) / d;
                        dx *= sFac;
                        dy *= sFac;
                        particlePosX[i]   -= dx;
                        particlePosY[i]   -= dy;
                        particlePosX[idn] += dx;
                        particlePosY[idn] += dy;

                        // diffuse colours
                        float c0r = particleColorR[i],  c1r = particleColorR[idn];
                        float c0g = particleColorG[i],  c1g = particleColorG[idn];
                        float c0b = particleColorB[i],  c1b = particleColorB[idn];
                        float cr = (c0r + c1r) * 0.5f;
                        float cg = (c0g + c1g) * 0.5f;
                        float cb = (c0b + c1b) * 0.5f;
                        particleColorR[i]   = c0r + (cr - c0r) * colorDiffusionCoeff;
                        particleColorR[idn] = c1r + (cr - c1r) * colorDiffusionCoeff;
                        particleColorG[i]   = c0g + (cg - c0g) * colorDiffusionCoeff;
                        particleColorG[idn] = c1g + (cg - c1g) * colorDiffusionCoeff;
                        particleColorB[i]   = c0b + (cb - c0b) * colorDiffusionCoeff;
                        particleColorB[idn] = c1b + (cb - c1b) * colorDiffusionCoeff;

                        // refresh cached px/py for next neighbour
                        px = particlePosX[i];
                        py = particlePosY[i];
                    }
                }
            }
        }
    }
}

void FlipFluid::handleParticleCollisions(float obstacleX, float obstacleY,
                                         float obstacleRadius,
                                         float obstacleVelX, float obstacleVelY)
{
    float hh = 1.0f / fInvSpacing;
    float r = particleRadius;
    float minDist = obstacleRadius + r;
    float minDist2 = minDist * minDist;

    float minX = hh + r;
    float maxX = (fNumX - 1) * hh - r;
    float minY = hh + r;
    float maxY = (fNumY - 1) * hh - r;

    for (int i = 0; i < numParticles; ++i) {
        float x = particlePosX[i];
        float y = particlePosY[i];

        float dx = x - obstacleX;
        float dy = y - obstacleY;
        float d2 = dx * dx + dy * dy;

        if (d2 < minDist2) {
            particleVelX[i] = obstacleVelX;
            particleVelY[i] = obstacleVelY;
        }

        if (x < minX) { x = minX; particleVelX[i] = 0.0f; }
        if (x > maxX) { x = maxX; particleVelX[i] = 0.0f; }
        if (y < minY) { y = minY; particleVelY[i] = 0.0f; }
        if (y > maxY) { y = maxY; particleVelY[i] = 0.0f; }

        particlePosX[i] = x;
        particlePosY[i] = y;
    }
}

void FlipFluid::updateParticleDensity() {
    int n = fNumY;
    float hh = h;
    float h1 = fInvSpacing;
    float h2 = 0.5f * h;

    std::fill(particleDensity.begin(), particleDensity.end(), 0.0f);

    for (int i = 0; i < numParticles; ++i) {
        float x = particlePosX[i];
        float y = particlePosY[i];

        x = clampf(x, hh, (fNumX - 1) * hh);
        y = clampf(y, hh, (fNumY - 1) * hh);

        int x0 = int(std::floor((x - h2) * h1));
        float tx = ((x - h2) - x0 * hh) * h1;
        int x1 = std::min(x0 + 1, fNumX - 2);

        int y0 = int(std::floor((y - h2) * h1));
        float ty = ((y - h2) - y0 * hh) * h1;
        int y1 = std::min(y0 + 1, fNumY - 2);

        float sx = 1.0f - tx;
        float sy = 1.0f - ty;

        if (x0 < fNumX && y0 < fNumY) particleDensity[x0 * n + y0] += sx * sy;
        if (x1 < fNumX && y0 < fNumY) particleDensity[x1 * n + y0] += tx * sy;
        if (x1 < fNumX && y1 < fNumY) particleDensity[x1 * n + y1] += tx * ty;
        if (x0 < fNumX && y1 < fNumY) particleDensity[x0 * n + y1] += sx * ty;
    }
}

float FlipFluid::computeRestDensity() {
    float sumD = 0.0f;
    int numFluidCells = 0;
    for (int i = 0; i < fNumCells; ++i) {
        if (cellType[i] == FLUID_CELL) {
            sumD += particleDensity[i];
            numFluidCells++;
        }
    }
    return (numFluidCells > 0) ? (sumD / numFluidCells) : 0.0f;
}

void FlipFluid::classifyCells() {
    for (int i = 0; i < fNumCells; ++i)
        cellType[i] = (s[i] == 0.0f) ? SOLID_CELL : AIR_CELL;
    for (int i = 0; i < numParticles; ++i) {
        int xi = clampi(int(std::floor(particlePosX[i] * fInvSpacing)), 0, fNumX - 1);
        int yi = clampi(int(std::floor(particlePosY[i] * fInvSpacing)), 0, fNumY - 1);
        int cellNr = xi * fNumY + yi;
        if (cellType[cellNr] == AIR_CELL)
            cellType[cellNr] = FLUID_CELL;
    }
}

void FlipFluid::savePrevVelocities() {
    for (int i = 0; i < fNumCells; ++i) {
        prevU[i] = u[i];
        prevV[i] = v[i];
        du[i] = 0.0f;
        dv[i] = 0.0f;
        u[i] = 0.0f;
        v[i] = 0.0f;
    }
}

void FlipFluid::p2gComponent(int component) {
    int n = fNumY;
    float hh = h;
    float h1 = fInvSpacing;
    float h2 = 0.5f * h;
    float dxOff = (component == 0) ? 0.0f : h2;
    float dyOff = (component == 0) ? h2   : 0.0f;

    auto* fld  = (component == 0) ? u.data()  : v.data();
    auto* fldD = (component == 0) ? du.data() : dv.data();

    for (int i = 0; i < numParticles; ++i) {
        float x = particlePosX[i];
        float y = particlePosY[i];

        x = clampf(x, hh, (fNumX - 1) * hh);
        y = clampf(y, hh, (fNumY - 1) * hh);

        int x0 = std::min(int(std::floor((x - dxOff) * h1)), fNumX - 2);
        float tx = ((x - dxOff) - x0 * hh) * h1;
        int x1 = std::min(x0 + 1, fNumX - 2);

        int y0 = std::min(int(std::floor((y - dyOff) * h1)), fNumY - 2);
        float ty = ((y - dyOff) - y0 * hh) * h1;
        int y1 = std::min(y0 + 1, fNumY - 2);

        float sx = 1.0f - tx;
        float sy = 1.0f - ty;

        float d0 = sx * sy;
        float d1 = tx * sy;
        float d2 = tx * ty;
        float d3 = sx * ty;

        int nr0 = x0 * n + y0;
        int nr1 = x1 * n + y0;
        int nr2 = x1 * n + y1;
        int nr3 = x0 * n + y1;

        float pv = (component == 0) ? particleVelX[i] : particleVelY[i];
        fld[nr0] += pv * d0; fldD[nr0] += d0;
        fld[nr1] += pv * d1; fldD[nr1] += d1;
        fld[nr2] += pv * d2; fldD[nr2] += d2;
        fld[nr3] += pv * d3; fldD[nr3] += d3;
    }
}

void FlipFluid::p2gNormalize(int component) {
    if (component == 0) {
        for (int i = 0; i < fNumCells; ++i)
            if (du[i] > 0.0f) u[i] /= du[i];
    } else {
        for (int i = 0; i < fNumCells; ++i)
            if (dv[i] > 0.0f) v[i] /= dv[i];
    }
}

void FlipFluid::restoreSolidCells() {
    int n = fNumY;
    for (int i = 0; i < fNumX; ++i) {
        for (int j = 0; j < fNumY; ++j) {
            bool solid = (cellType[i * n + j] == SOLID_CELL);
            if (solid || (i > 0 && cellType[(i - 1) * n + j] == SOLID_CELL))
                u[i * n + j] = prevU[i * n + j];
            if (solid || (j > 0 && cellType[i * n + j - 1] == SOLID_CELL))
                v[i * n + j] = prevV[i * n + j];
        }
    }
}

void FlipFluid::g2pComponent(int component, float flipRatio) {
    int n = fNumY;
    float hh = h;
    float h1 = fInvSpacing;
    float h2 = 0.5f * h;
    float dxOff = (component == 0) ? 0.0f : h2;
    float dyOff = (component == 0) ? h2   : 0.0f;
    int offset = (component == 0) ? n : 1;

    const float* fld  = (component == 0) ? u.data()      : v.data();
    const float* pfld = (component == 0) ? prevU.data()  : prevV.data();

    for (int i = 0; i < numParticles; ++i) {
        float x = particlePosX[i];
        float y = particlePosY[i];

        x = clampf(x, hh, (fNumX - 1) * hh);
        y = clampf(y, hh, (fNumY - 1) * hh);

        int x0 = std::min(int(std::floor((x - dxOff) * h1)), fNumX - 2);
        float tx = ((x - dxOff) - x0 * hh) * h1;
        int x1 = std::min(x0 + 1, fNumX - 2);

        int y0 = std::min(int(std::floor((y - dyOff) * h1)), fNumY - 2);
        float ty = ((y - dyOff) - y0 * hh) * h1;
        int y1 = std::min(y0 + 1, fNumY - 2);

        float sx = 1.0f - tx;
        float sy = 1.0f - ty;

        float d0 = sx * sy;
        float d1 = tx * sy;
        float d2 = tx * ty;
        float d3 = sx * ty;

        int nr0 = x0 * n + y0;
        int nr1 = x1 * n + y0;
        int nr2 = x1 * n + y1;
        int nr3 = x0 * n + y1;

        float valid0 = (cellType[nr0] != AIR_CELL || cellType[nr0 - offset] != AIR_CELL) ? 1.0f : 0.0f;
        float valid1 = (cellType[nr1] != AIR_CELL || cellType[nr1 - offset] != AIR_CELL) ? 1.0f : 0.0f;
        float valid2 = (cellType[nr2] != AIR_CELL || cellType[nr2 - offset] != AIR_CELL) ? 1.0f : 0.0f;
        float valid3 = (cellType[nr3] != AIR_CELL || cellType[nr3 - offset] != AIR_CELL) ? 1.0f : 0.0f;

        float v_old = (component == 0) ? particleVelX[i] : particleVelY[i];
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
            if (component == 0) particleVelX[i] = blended;
            else                particleVelY[i] = blended;
        }
    }
}

void FlipFluid::transferVelocities(bool toGrid, float flipRatio) {
    if (toGrid) {
        savePrevVelocities();
        classifyCells();
        p2gComponent(0);
        p2gComponent(1);
        p2gNormalize(0);
        p2gNormalize(1);
        restoreSolidCells();
    } else {
        g2pComponent(0, flipRatio);
        g2pComponent(1, flipRatio);
    }
}

void FlipFluid::solveIncompressibility(int numIters, float dt,
                                       float overRelaxation, bool compensateDrift)
{
    // zero pressure, save prev velocities
    for (int i = 0; i < fNumCells; ++i) {
        p[i] = 0.0f;
        prevU[i] = u[i];
        prevV[i] = v[i];
    }

    int n = fNumY;
    float cp = density * h / dt;
    float rest = particleRestDensity;
    int   cd = compensateDrift ? 1 : 0;

    for (int iter = 0; iter < numIters; ++iter) {
        // Gauss-Seidel sweep matching the JS ordering exactly: i in [1, fNumX-1),
        // j in [1, fNumY-1).
        for (int i = 1; i < fNumX - 1; ++i) {
            for (int j = 1; j < fNumY - 1; ++j) {
                if (cellType[i * n + j] != FLUID_CELL) continue;

                int center = i * n + j;
                int left   = (i - 1) * n + j;
                int right  = (i + 1) * n + j;
                int bottom = i * n + j - 1;
                int top    = i * n + j + 1;

                float sx0 = s[left];
                float sx1 = s[right];
                float sy0 = s[bottom];
                float sy1 = s[top];
                float sSum = sx0 + sx1 + sy0 + sy1;
                if (sSum == 0.0f) continue;

                float div = u[right] - u[center] + v[top] - v[center];

                if (rest > 0.0f && cd != 0) {
                    float k = 1.0f;
                    float compression = particleDensity[center] - rest;
                    if (compression > 0.0f)
                        div = div - k * compression;
                }

                float pVal = -div / sSum;
                pVal *= overRelaxation;
                p[center] += cp * pVal;

                u[center] -= sx0 * pVal;
                u[right]  += sx1 * pVal;
                v[center] -= sy0 * pVal;
                v[top]    += sy1 * pVal;
            }
        }
    }
}

void FlipFluid::updateParticleColors() {
    const float sStep = 0.01f;
    float d0 = particleRestDensity;
    float h1 = fInvSpacing;

    for (int i = 0; i < numParticles; ++i) {
        particleColorR[i] = clampf(particleColorR[i] - sStep, 0.0f, 1.0f);
        particleColorG[i] = clampf(particleColorG[i] - sStep, 0.0f, 1.0f);
        particleColorB[i] = clampf(particleColorB[i] + sStep, 0.0f, 1.0f);

        int xi = clampi(int(std::floor(particlePosX[i] * h1)), 1, fNumX - 1);
        int yi = clampi(int(std::floor(particlePosY[i] * h1)), 1, fNumY - 1);
        int cellNr = xi * fNumY + yi;

        if (d0 > 0.0f) {
            float relDensity = particleDensity[cellNr] / d0;
            if (relDensity < 0.7f) {
                float s2 = 0.8f;
                particleColorR[i] = s2;
                particleColorG[i] = s2;
                particleColorB[i] = 1.0f;
            }
        }
    }
}

void FlipFluid::setSciColor(int cellNr, float val, float minVal, float maxVal) {
    val = std::min(std::max(val, minVal), maxVal - 0.0001f);
    float d = maxVal - minVal;
    if (d == 0.0f) val = 0.5f;
    else           val = (val - minVal) / d;
    float m = 0.25f;
    int num = int(std::floor(val / m));
    float sLoc = (val - num * m) / m;
    float r = 0, g = 0, b = 0;
    switch (num) {
        case 0: r = 0.0f;     g = sLoc;        b = 1.0f;        break;
        case 1: r = 0.0f;     g = 1.0f;        b = 1.0f - sLoc; break;
        case 2: r = sLoc;     g = 1.0f;        b = 0.0f;        break;
        default: r = 1.0f;    g = 1.0f - sLoc; b = 0.0f;        break;
    }
    cellColor[3 * cellNr]     = r;
    cellColor[3 * cellNr + 1] = g;
    cellColor[3 * cellNr + 2] = b;
}

void FlipFluid::updateCellColors() {
    std::fill(cellColor.begin(), cellColor.end(), 0.0f);

    for (int i = 0; i < fNumCells; ++i) {
        if (cellType[i] == SOLID_CELL) {
            cellColor[3 * i]     = 0.5f;
            cellColor[3 * i + 1] = 0.5f;
            cellColor[3 * i + 2] = 0.5f;
        } else if (cellType[i] == FLUID_CELL) {
            float d = particleDensity[i];
            if (particleRestDensity > 0.0f)
                d /= particleRestDensity;
            setSciColor(i, d, 0.0f, 2.0f);
        }
    }
}

void FlipFluid::simulate(float dt, float gravity, float flipRatio,
                         int numPressureIters, int numParticleIters,
                         float overRelaxation, bool compensateDrift,
                         bool separateParticles,
                         float obstacleX, float obstacleY, float obstacleRadius,
                         float obstacleVelX, float obstacleVelY,
                         int numSubSteps)
{
    if (numSubSteps < 1) numSubSteps = 1;
    float sdt = dt / numSubSteps;
    for (int step = 0; step < numSubSteps; ++step) {
        integrateParticles(sdt, gravity);
        if (separateParticles)
            pushParticlesApart(numParticleIters);
        handleParticleCollisions(obstacleX, obstacleY, obstacleRadius,
                                 obstacleVelX, obstacleVelY);
        transferVelocities(true);
        updateParticleDensity();
        if (particleRestDensity == 0.0f)
            particleRestDensity = computeRestDensity();
        solveIncompressibility(numPressureIters, sdt, overRelaxation, compensateDrift);
        transferVelocities(false, flipRatio);
    }
    updateParticleColors();
    updateCellColors();
}

} // namespace flipcpu
