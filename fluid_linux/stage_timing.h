// Shared per-stage timing for the FLIP demo (CPU and CUDA builds).
//
// The pipeline of FlipFluid::simulate(...) is split into the fixed stages T1..T10
// (plus T_total) defined by the assignment. Both the CPU build (std::chrono) and
// the CUDA build (cudaEvent) accumulate per-frame milliseconds into a StageStats,
// then print an averaged table every N frames via stageReportEvery().

#pragma once

#include <cstdio>

namespace fliptiming {

enum Stage {
    T1_integrate = 0, // integrateParticles
    T2_pushApart,     // pushParticlesApart (histogram + scan + scatter + separation)
    T3_collisions,    // handleParticleCollisions
    T4_p2g,           // transferVelocities(toGrid=true): savePrev+classify+p2g+normalize+restoreSolid
    T5_density,       // updateParticleDensity (+ one-time computeRestDensity)
    T6_pressure,      // solveIncompressibility (pressure solve loop)
    T7_g2p,           // transferVelocities(toGrid=false)
    T8_colors,        // updateParticleColors + updateCellColors
    T9_render,        // draw + buffer swap (host GL)
    T10_transfer,     // CUDA only: D2H copy (no interop) or map/unmap (B1 interop). 0 on CPU.
    T_total,          // whole frame
    STAGE_COUNT
};

inline const char* stageName(int s) {
    static const char* names[STAGE_COUNT] = {
        "T1_integrate", "T2_pushApart", "T3_collisions", "T4_p2g",
        "T5_density",   "T6_pressure",  "T7_g2p",        "T8_colors",
        "T9_render",    "T10_transfer", "T_total"
    };
    return names[s];
}

struct StageStats {
    double sumMs[STAGE_COUNT] = {};
    int    frames = 0;
    int    lastPressureIters = 0;
};

inline void stageAdd(StageStats& st, Stage s, double ms) { st.sumMs[s] += ms; }
inline void stageSetIters(StageStats& st, int iters) { st.lastPressureIters = iters; }

// Call once per frame. When `frames` reaches N, print the averaged table and reset.
inline void stageReportEvery(StageStats& st, int N, const char* tag) {
    st.frames += 1;
    if (st.frames < N) return;

    double inv   = 1.0 / st.frames;
    double total = st.sumMs[T_total] * inv;

    std::printf("\n=== [%s] per-stage average over %d frames "
                "(numPressureIters=%d) ===\n",
                tag, st.frames, st.lastPressureIters);
    for (int s = 0; s < STAGE_COUNT; ++s) {
        double avg = st.sumMs[s] * inv;
        if (s == T_total) {
            std::printf("  %-13s %8.3f ms   (%6.1f FPS)\n",
                        stageName(s), avg, avg > 0.0 ? 1000.0 / avg : 0.0);
        } else {
            double pct = (total > 0.0) ? (avg / total * 100.0) : 0.0;
            std::printf("  %-13s %8.3f ms   %5.1f%%\n", stageName(s), avg, pct);
        }
    }
    std::fflush(stdout);

    for (int s = 0; s < STAGE_COUNT; ++s) st.sumMs[s] = 0.0;
    st.frames = 0;
}

}
