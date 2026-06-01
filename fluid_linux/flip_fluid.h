
#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace flipcpu {

constexpr int U_FIELD = 0;
constexpr int V_FIELD = 1;

constexpr int FLUID_CELL = 0;
constexpr int AIR_CELL   = 1;
constexpr int SOLID_CELL = 2;

class FlipFluid {
public:
    // Grid
    float density;
    int   fNumX;
    int   fNumY;
    float h;
    float fInvSpacing;
    int   fNumCells;

    std::vector<float> u, v, du, dv, prevU, prevV, p, s;
    std::vector<int>   cellType;
    std::vector<float> cellColor;        // 3 * fNumCells

    // Particles
    int   maxParticles;
    std::vector<float> particlePosX, particlePosY;
    std::vector<float> particleVelX, particleVelY;
    std::vector<float> particleColorR, particleColorG, particleColorB;
    std::vector<float> particleDensity; // fNumCells
    float particleRestDensity = 0.0f;

    float particleRadius;
    float pInvSpacing;
    int   pNumX;
    int   pNumY;
    int   pNumCells;

    std::vector<int> numCellParticles;
    std::vector<int> firstCellParticle;   // pNumCells + 1
    std::vector<int> cellParticleIds;     // maxParticles

    int  numParticles = 0;

    FlipFluid(float density, float width, float height,
              float spacing, float particle_radius, int max_particles);

    void integrateParticles(float dt, float gravity);
    void pushParticlesApart(int numIters);
    void handleParticleCollisions(float obstacleX, float obstacleY,
                                  float obstacleRadius,
                                  float obstacleVelX, float obstacleVelY);
    void updateParticleDensity();
    void transferVelocities(bool toGrid, float flipRatio = 0.0f);
    void solveIncompressibility(int numIters, float dt,
                                float overRelaxation, bool compensateDrift);
    void updateParticleColors();
    void updateCellColors();

    void simulate(float dt, float gravity, float flipRatio,
                  int numPressureIters, int numParticleIters,
                  float overRelaxation, bool compensateDrift,
                  bool separateParticles,
                  float obstacleX, float obstacleY, float obstacleRadius,
                  float obstacleVelX, float obstacleVelY,
                  int numSubSteps = 1);

private:
    void p2gComponent(int component);
    void p2gNormalize(int component);
    void g2pComponent(int component, float flipRatio);
    void classifyCells();
    void savePrevVelocities();
    void restoreSolidCells();
    float computeRestDensity();
    void setSciColor(int cellNr, float val, float minVal, float maxVal);
};

} // namespace flipcpu
