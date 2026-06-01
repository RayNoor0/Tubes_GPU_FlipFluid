// Per-stage timing + benchmark CSV emission shared by the CPU and CUDA builds.
//
// The pipeline stage codes mirror the assignment exactly (T1..T10, T_total).
// Both builds feed per-stage milliseconds into a StageTimers instance: the CPU
// build via std::chrono, the CUDA build via cudaEvent elapsed time. Timings are
// accumulated per frame (so sub-steps within one simulate() sum together), then
// averaged over the measured window.
//
// Header-only and pure C++ so it compiles cleanly under both g++ and nvcc.

#pragma once

#include <cstdio>
#include <cstring>
#include <string>

namespace flipbench {

enum Stage {
    T1_integrate = 0,
    T2_pushApart,
    T3_collisions,
    T4_p2g,
    T5_density,
    T6_pressure,
    T7_g2p,
    T8_colors,
    T9_render,
    T10_h2d,
    T10_d2h,
    NUM_STAGES
};

inline const char* stageName(int s) {
    switch (s) {
        case T1_integrate:  return "T1_integrate";
        case T2_pushApart:  return "T2_pushApart";
        case T3_collisions: return "T3_collisions";
        case T4_p2g:        return "T4_p2g";
        case T5_density:    return "T5_density";
        case T6_pressure:   return "T6_pressure";
        case T7_g2p:        return "T7_g2p";
        case T8_colors:     return "T8_colors";
        case T9_render:     return "T9_render";
        case T10_h2d:       return "T10_h2d";
        case T10_d2h:       return "T10_d2h";
        default:            return "?";
    }
}

struct StageTimers {
    double accum[NUM_STAGES];   // summed ms over all measured frames
    double frame[NUM_STAGES];   // ms accumulated in the in-progress frame
    double totalAccum;          // summed T_total ms over measured frames
    double frameTotal;          // T_total for the in-progress frame
    long   framesMeasured;
    bool   measuring;           // false during warm-up

    // Context captured for the CSV row.
    int  resolution;
    int  numParticles;
    int  numPressureIters;

    StageTimers() { reset(); }

    void reset() {
        std::memset(accum, 0, sizeof(accum));
        std::memset(frame, 0, sizeof(frame));
        totalAccum = 0.0;
        frameTotal = 0.0;
        framesMeasured = 0;
        measuring = false;
        resolution = 0;
        numParticles = 0;
        numPressureIters = 0;
    }

    // Add milliseconds to a stage of the in-progress frame.
    void add(int stage, double ms) { frame[stage] += ms; }

    // Close the current frame: roll the frame buffer into the running totals
    // (only when measuring) and clear it for the next frame.
    void endFrame() {
        if (measuring) {
            for (int i = 0; i < NUM_STAGES; ++i) accum[i] += frame[i];
            // T_total is the sum of stages by default; callers may override
            // frameTotal explicitly (whole-frame wall clock) before endFrame().
            if (frameTotal <= 0.0) {
                double s = 0.0;
                for (int i = 0; i < NUM_STAGES; ++i) s += frame[i];
                frameTotal = s;
            }
            totalAccum += frameTotal;
            ++framesMeasured;
        }
        std::memset(frame, 0, sizeof(frame));
        frameTotal = 0.0;
    }

    double meanMs(int stage) const {
        return framesMeasured > 0 ? accum[stage] / framesMeasured : 0.0;
    }
    double meanTotalMs() const {
        return framesMeasured > 0 ? totalAccum / framesMeasured : 0.0;
    }

    // ----- CSV output -----------------------------------------------------
    static void writeCsvHeader(std::FILE* f) {
        std::fprintf(f, "build,resolution,numParticles,numPressureIters");
        for (int i = 0; i < NUM_STAGES; ++i)
            std::fprintf(f, ",%s", stageName(i));
        std::fprintf(f, ",T_total\n");
    }

    void writeCsvRow(std::FILE* f, const char* build) const {
        std::fprintf(f, "%s,%d,%d,%d", build, resolution,
                     numParticles, numPressureIters);
        for (int i = 0; i < NUM_STAGES; ++i)
            std::fprintf(f, ",%.6f", meanMs(i));
        std::fprintf(f, ",%.6f\n", meanTotalMs());
    }

    void printSummary(const char* build) const {
        std::printf("[%s] res=%d particles=%d pIters=%d frames=%ld\n",
                    build, resolution, numParticles, numPressureIters,
                    framesMeasured);
        for (int i = 0; i < NUM_STAGES; ++i)
            std::printf("    %-14s %8.4f ms\n", stageName(i), meanMs(i));
        std::printf("    %-14s %8.4f ms\n", "T_total", meanTotalMs());
    }
};

} // namespace flipbench
