// Window + renderer + main loop for the CUDA FLIP demo.
// The simulation runs on the GPU (flipcuda::FlipFluidCuda). Particle positions and
// colors live in OpenGL VBOs that CUDA maps each frame and writes directly (bonus
// B1 interop) — there is no device->host copy on the render path. Per-stage timing
// (T1..T10, T_total) prints every 60 frames, matching the CPU build's format.
//
// Inputs (identical to the CPU demo):
//   left mouse drag   move/release the obstacle
//   SPACE / P         pause-resume
//   G                 toggle grid
//   R                 reset scene
//   Q / Esc           quit

#include "flip_fluid_cuda.cuh"
#include "../ui.h"

#include <GL/glext.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/glx.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace flipcuda;

// --------------------------- scene / config ---------------------------------
struct Scene {
    float gravity         = -9.81f;
    float dt              = 1.0f / 60.0f;
    float flipRatio       = 0.9f;
    int   numPressureIters= 50;
    int   numParticleIters= 2;
    long  frameNr         = 0;
    float overRelaxation  = 1.9f;
    bool  compensateDrift = true;
    bool  separateParticles = true;
    float obstacleX       = 0.0f;
    float obstacleY       = 0.0f;
    float obstacleRadius  = 0.15f;
    bool  paused          = true;
    bool  showObstacle    = true;
    float obstacleVelX    = 0.0f;
    float obstacleVelY    = 0.0f;
    bool  showParticles   = true;
    bool  showGrid        = false;
    int   resolution      = 100;
    int   numSubSteps     = 1;
    FlipFluidCuda* fluid  = nullptr;
};

static Scene scene;

constexpr int CANVAS_W = 900;
constexpr int CANVAS_H = 700;
constexpr float simHeight = 3.0f;
constexpr float cScale = float(CANVAS_H) / simHeight;
constexpr float simWidth = float(CANVAS_W) / cScale;

// --------------------------- GL VBO function pointers ------------------------
static PFNGLGENBUFFERSPROC    pglGenBuffers    = nullptr;
static PFNGLBINDBUFFERPROC    pglBindBuffer    = nullptr;
static PFNGLBUFFERDATAPROC    pglBufferData    = nullptr;
static PFNGLDELETEBUFFERSPROC pglDeleteBuffers = nullptr;

static void loadGLBuffers() {
    pglGenBuffers    = (PFNGLGENBUFFERSPROC)    glXGetProcAddressARB((const GLubyte*)"glGenBuffers");
    pglBindBuffer    = (PFNGLBINDBUFFERPROC)    glXGetProcAddressARB((const GLubyte*)"glBindBuffer");
    pglBufferData    = (PFNGLBUFFERDATAPROC)    glXGetProcAddressARB((const GLubyte*)"glBufferData");
    pglDeleteBuffers = (PFNGLDELETEBUFFERSPROC) glXGetProcAddressARB((const GLubyte*)"glDeleteBuffers");
    if (!pglGenBuffers || !pglBindBuffer || !pglBufferData || !pglDeleteBuffers) {
        std::fprintf(stderr, "Failed to load GL VBO functions (need GL >= 1.5)\n");
        std::exit(1);
    }
}

static GLuint g_posVBO = 0;   // float2 per particle (interop)
static GLuint g_colVBO = 0;   // float3 per particle (interop)

// --------------------------- scene helpers ----------------------------------
static void setObstacle(float x, float y, bool reset) {
    float vx = 0.0f, vy = 0.0f;
    if (!reset) {
        vx = (x - scene.obstacleX) / scene.dt;
        vy = (y - scene.obstacleY) / scene.dt;
    }
    scene.obstacleX = x;
    scene.obstacleY = y;
    scene.fluid->carveObstacle(x, y, scene.obstacleRadius, vx, vy);
    scene.showObstacle  = true;
    scene.obstacleVelX  = vx;
    scene.obstacleVelY  = vy;
}

// Mirror of the CPU seedParticles, producing interleaved host arrays for upload.
static void seedParticles(std::vector<float>& posXY, std::vector<float>& colRGB,
                          int numX, int numY, float h, float r, float dx, float dy) {
    for (int i = 0; i < numX; ++i) {
        for (int j = 0; j < numY; ++j) {
            int pid = i * numY + j;
            float offset = (j % 2 == 0) ? 0.0f : r;
            posXY[2 * pid + 0] = h + r + dx * i + offset;
            posXY[2 * pid + 1] = h + r + dy * j;
            colRGB[3 * pid + 0] = 0.0f;
            colRGB[3 * pid + 1] = 0.0f;
            colRGB[3 * pid + 2] = 1.0f;
        }
    }
}

static void setupTank(std::vector<float>& s, int fNumX, int fNumY) {
    int n = fNumY;
    for (int i = 0; i < fNumX; ++i)
        for (int j = 0; j < fNumY; ++j) {
            float sVal = 1.0f;
            if (i == 0 || i == fNumX - 1 || j == 0) sVal = 0.0f;
            s[i * n + j] = sVal;
        }
}

static void setupScene() {
    scene.obstacleRadius   = 0.15f;
    // Pressure solver is red-black Gauss-Seidel (see solvePressure) — stable at the
    // CPU's SOR factor, so we keep overRelaxation = 1.9 and numPressureIters identical
    // to the CPU for an apples-to-apples comparison (only update ordering differs).
    scene.overRelaxation   = 1.9f;
    scene.dt               = 1.0f / 60.0f;
    scene.numParticleIters = 2;

    int res = scene.resolution;
    if      (res <= 100) scene.numSubSteps = 1;
    else if (res <= 140) scene.numSubSteps = 2;
    else if (res <= 180) scene.numSubSteps = 3;
    else                 scene.numSubSteps = 4;
    scene.numPressureIters = 50 + std::max(0, (res - 100)) / 2;

    float tankHeight = 1.0f * simHeight;
    float tankWidth  = 1.0f * simWidth;
    float h          = tankHeight / res;
    float density    = 1000.0f;

    float relWaterHeight = 0.8f;
    float relWaterWidth  = 0.6f;
    float r  = 0.3f * h;
    float dx = 2.0f * r;
    float dy = std::sqrt(3.0f) / 2.0f * dx;

    int numX = int(std::floor((relWaterWidth  * tankWidth  - 2.0f * h - 2.0f * r) / dx));
    int numY = int(std::floor((relWaterHeight * tankHeight - 2.0f * h - 2.0f * r) / dy));
    if (numX < 1) numX = 1;
    if (numY < 1) numY = 1;
    int maxParticles = numX * numY;

    // Tear down the previous fluid first so its CUDA resources are unregistered
    // before the underlying VBOs are deleted.
    delete scene.fluid;
    scene.fluid = nullptr;

    // (Re)create the interop VBOs sized for this resolution.
    if (g_posVBO) pglDeleteBuffers(1, &g_posVBO);
    if (g_colVBO) pglDeleteBuffers(1, &g_colVBO);
    pglGenBuffers(1, &g_posVBO);
    pglGenBuffers(1, &g_colVBO);

    std::vector<float> hostPos(2 * maxParticles, 0.0f);
    std::vector<float> hostCol(3 * maxParticles, 0.0f);
    seedParticles(hostPos, hostCol, numX, numY, h, r, dx, dy);

    pglBindBuffer(GL_ARRAY_BUFFER, g_posVBO);
    pglBufferData(GL_ARRAY_BUFFER, hostPos.size() * sizeof(float),
                  hostPos.data(), GL_DYNAMIC_DRAW);
    pglBindBuffer(GL_ARRAY_BUFFER, g_colVBO);
    pglBufferData(GL_ARRAY_BUFFER, hostCol.size() * sizeof(float),
                  hostCol.data(), GL_DYNAMIC_DRAW);
    pglBindBuffer(GL_ARRAY_BUFFER, 0);

    scene.fluid = new FlipFluidCuda(density, tankWidth, tankHeight, h, r, maxParticles);
    FlipFluidCuda& f = *scene.fluid;
    f.numParticles = numX * numY;
    f.uploadParticles(hostPos, hostCol);
    bool interop = f.tryRegisterVBOs(g_posVBO, g_colVBO);
    std::printf("[flip-cuda] CUDA-GL interop: %s\n",
                interop ? "ENABLED (B1, kernels write VBO directly)"
                        : "DISABLED -> device->host copy (T10_d2h) [expected on WSL2]");

    std::vector<float> s(f.fNumCells, 0.0f);
    setupTank(s, f.fNumX, f.fNumY);
    f.uploadSolid(s);

    setObstacle(3.0f, 2.0f, true);
    f.updateCellColorsOnce();
    scene.frameNr = 0;
}

// --------------------------- X11 + GLX --------------------------------------
static int s_glxAttrs[] = {
    GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 24,
    GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
    None
};

struct AppWindow {
    Display* dpy = nullptr;
    ::Window xwin = 0;
    GLXContext glc = nullptr;
    XVisualInfo* vi = nullptr;
    Atom wm_delete = 0;
    int width = CANVAS_W;
    int height = CANVAS_H;
    bool running = true;
};

static bool createWindow(AppWindow& w, const char* title) {
    w.dpy = XOpenDisplay(nullptr);
    if (!w.dpy) { std::fprintf(stderr, "Cannot open X display\n"); return false; }
    int screen = DefaultScreen(w.dpy);
    w.vi = glXChooseVisual(w.dpy, screen, s_glxAttrs);
    if (!w.vi) { std::fprintf(stderr, "No suitable GLX visual\n"); return false; }
    ::Window root = RootWindow(w.dpy, screen);
    Colormap cmap = XCreateColormap(w.dpy, root, w.vi->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.colormap = cmap;
    swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask |
                     ButtonPressMask | ButtonReleaseMask |
                     PointerMotionMask | StructureNotifyMask;
    w.xwin = XCreateWindow(w.dpy, root, 0, 0, w.width, w.height, 0,
                           w.vi->depth, InputOutput, w.vi->visual,
                           CWColormap | CWEventMask, &swa);
    XStoreName(w.dpy, w.xwin, title);
    w.wm_delete = XInternAtom(w.dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(w.dpy, w.xwin, &w.wm_delete, 1);
    XMapWindow(w.dpy, w.xwin);

    w.glc = glXCreateContext(w.dpy, w.vi, nullptr, GL_TRUE);
    if (!w.glc) { std::fprintf(stderr, "glXCreateContext failed\n"); return false; }
    glXMakeCurrent(w.dpy, w.xwin, w.glc);

    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    return true;
}

static void destroyWindow(AppWindow& w) {
    if (w.glc) { glXMakeCurrent(w.dpy, None, nullptr); glXDestroyContext(w.dpy, w.glc); }
    if (w.xwin) XDestroyWindow(w.dpy, w.xwin);
    if (w.dpy)  XCloseDisplay(w.dpy);
}

// --------------------------- rendering --------------------------------------
static void setProjection(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, simWidth, 0.0, simHeight, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Grid overlay (debug, default off): pulls cellColor back to the host. This is the
// only device->host copy in the program and is skipped unless the grid is shown.
static void drawGrid(FlipFluidCuda& f) {
    static std::vector<float> cc;
    f.downloadCellColors(cc);
    float h = f.h;
    glBegin(GL_QUADS);
    for (int i = 0; i < f.fNumX; ++i) {
        for (int j = 0; j < f.fNumY; ++j) {
            int idx = i * f.fNumY + j;
            glColor3f(cc[3 * idx + 0], cc[3 * idx + 1], cc[3 * idx + 2]);
            float x0 = i * h, y0 = j * h, x1 = x0 + h, y1 = y0 + h;
            glVertex2f(x0, y0); glVertex2f(x1, y0);
            glVertex2f(x1, y1); glVertex2f(x0, y1);
        }
    }
    glEnd();
}

static void drawParticles(FlipFluidCuda& f, int viewportH) {
    float pxPerSimUnit = float(viewportH) / simHeight;
    float diameterPx = 2.0f * f.particleRadius * pxPerSimUnit;
    if (diameterPx < 1.0f) diameterPx = 1.0f;
    glPointSize(diameterPx);

    if (f.usingInterop()) {
        // Kernels wrote the VBOs directly (B1) — draw straight from them.
        pglBindBuffer(GL_ARRAY_BUFFER, g_posVBO);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 0, (void*)0);
        pglBindBuffer(GL_ARRAY_BUFFER, g_colVBO);
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_FLOAT, 0, (void*)0);
        glDrawArrays(GL_POINTS, 0, f.numParticles);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        pglBindBuffer(GL_ARRAY_BUFFER, 0);
    } else {
        // No interop: draw from the host arrays filled by the D2H copy in simulate().
        pglBindBuffer(GL_ARRAY_BUFFER, 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 0, f.hostPositions().data());
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_FLOAT, 0, f.hostColors().data());
        glDrawArrays(GL_POINTS, 0, f.numParticles);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    }
}

static void drawObstacle(FlipFluidCuda& f, float ox, float oy, float orad) {
    const int N = 48;
    float drawR = orad + f.particleRadius;
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(ox, oy);
    for (int i = 0; i <= N; ++i) {
        float a = (float)i / N * 2.0f * (float)M_PI;
        glVertex2f(ox + drawR * std::cos(a), oy + drawR * std::sin(a));
    }
    glEnd();
}

// --------------------------- benchmark ----------------------
static void printGpuInfo() {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return;
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    int memClkKHz = 0, busBits = 0;
    cudaDeviceGetAttribute(&memClkKHz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busBits,  cudaDevAttrGlobalMemoryBusWidth, dev);
    double bwGBs = 2.0 * (double)memClkKHz * 1e3 * (busBits / 8.0) / 1e9; // DDR x2
    std::printf("[flip-cuda] GPU: %s | compute capability %d.%d | %.1f GiB VRAM | SMs=%d\n",
                p.name, p.major, p.minor,
                p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), p.multiProcessorCount);
    std::printf("[flip-cuda] mem: %.0f MHz x %d-bit -> %.1f GB/s peak (from device attrs)\n",
                memClkKHz / 1000.0, busBits, bwGBs);
    std::printf("[flip-cuda] CUDA driver %d.%d, runtime %d.%d\n",
                drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);
}

static void runBenchmark(AppWindow& w) {
    static const int resList[] = {50, 100, 150, 200};
    const int NRES = (int)(sizeof(resList) / sizeof(resList[0]));
    const int WARMUP = 60, MEASURE = 600;

    double avg[4][fliptiming::STAGE_COUNT] = {};
    int parts[4] = {}, pIters[4] = {}, subs[4] = {};

    auto now = []{ return std::chrono::steady_clock::now(); };
    auto ms  = [](std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    printGpuInfo();
    std::printf("[flip-cuda] benchmark: %d warmup + %d measured frames/resolution, vsync off\n",
                WARMUP, MEASURE);

    bool interop = false;
    for (int ri = 0; ri < NRES && w.running; ++ri) {
        scene.resolution = resList[ri];
        setupScene();
        scene.paused = false;
        FlipFluidCuda& f = *scene.fluid;
        interop = f.usingInterop();
        parts[ri] = f.numParticles; pIters[ri] = scene.numPressureIters;
        subs[ri]  = scene.numSubSteps;
        std::printf("  res=%-3d particles=%-7d pressIters=%-3d subSteps=%d ... ",
                    resList[ri], parts[ri], pIters[ri], subs[ri]);
        std::fflush(stdout);

        for (int frame = 0; frame < WARMUP + MEASURE && w.running; ++frame) {
            while (XPending(w.dpy) > 0) {
                XEvent e; XNextEvent(w.dpy, &e);
                if (e.type == ClientMessage &&
                    (Atom)e.xclient.data.l[0] == w.wm_delete) w.running = false;
                else if (e.type == ConfigureNotify) {
                    w.width = e.xconfigure.width; w.height = e.xconfigure.height;
                } else if (e.type == KeyPress) {
                    KeySym ks = XLookupKeysym(&e.xkey, 0);
                    if (ks == XK_q || ks == XK_Q || ks == XK_Escape) w.running = false;
                }
            }

            auto t0 = now();
            f.simulate(scene.dt, scene.gravity, scene.flipRatio,
                       scene.numPressureIters, scene.numParticleIters,
                       scene.overRelaxation, scene.compensateDrift,
                       scene.separateParticles,
                       scene.obstacleX, scene.obstacleY, scene.obstacleRadius,
                       scene.obstacleVelX, scene.obstacleVelY, scene.numSubSteps);
            auto tr = now();
            glClear(GL_COLOR_BUFFER_BIT);
            setProjection(w.width, w.height);
            if (scene.showGrid)      drawGrid(f);
            if (scene.showParticles) drawParticles(f, w.height);
            if (scene.showObstacle)  drawObstacle(f, scene.obstacleX, scene.obstacleY,
                                                  scene.obstacleRadius);
            glXSwapBuffers(w.dpy, w.xwin);
            auto te = now();

            fliptiming::stageAdd(f.timing, fliptiming::T9_render, ms(tr, te));
            fliptiming::stageAdd(f.timing, fliptiming::T_total,   ms(t0, te));
            if (frame == WARMUP - 1)
                for (int s = 0; s < fliptiming::STAGE_COUNT; ++s) f.timing.sumMs[s] = 0.0;
        }
        for (int s = 0; s < fliptiming::STAGE_COUNT; ++s)
            avg[ri][s] = f.timing.sumMs[s] / MEASURE;
        std::printf("done\n");
    }

    std::printf("\n=== [CUDA] FLIP per-stage benchmark (avg ms/frame) ===\n");
    std::printf("warmup=%d, measure=%d frames, vsync OFF, UI overlay excluded from T9\n",
                WARMUP, MEASURE);
    std::printf("render path: %s (T10 = %s)\n",
                interop ? "CUDA-GL interop (B1)" : "no-interop device->host copy",
                interop ? "map/unmap" : "D2H copy");
    std::printf("config: gravity ON, separateParticles ON, compensateDrift ON, "
                "flipRatio=%.2f, obstacle static @ (%.1f,%.1f); pressure = red-black GS\n\n",
                scene.flipRatio, scene.obstacleX, scene.obstacleY);
    std::printf("%-16s", "Stage");
    for (int ri = 0; ri < NRES; ++ri) std::printf("res=%-8d", resList[ri]);
    std::printf("\n");
    for (int s = 0; s < fliptiming::STAGE_COUNT; ++s) {
        std::printf("%-16s", fliptiming::stageName(s));
        for (int ri = 0; ri < NRES; ++ri) std::printf("%-11.4f", avg[ri][s]);
        std::printf("\n");
    }
    auto rowI = [&](const char* label, const int* v) {
        std::printf("%-16s", label);
        for (int ri = 0; ri < NRES; ++ri) std::printf("%-11d", v[ri]);
        std::printf("\n");
    };
    std::printf("%-16s", "FPS(1000/Ttot)");
    for (int ri = 0; ri < NRES; ++ri) {
        double t = avg[ri][fliptiming::T_total];
        std::printf("%-11.1f", t > 0.0 ? 1000.0 / t : 0.0);
    }
    std::printf("\n");
    rowI("particles", parts);
    rowI("pressIters", pIters);
    rowI("subSteps", subs);
    std::fflush(stdout);
}

// --------------------------- main -------------------------------------------
int main(int argc, char** argv) {
    bool noVsync = false;
    bool bench   = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-vsync") == 0) noVsync = true;
        else if (std::strcmp(argv[i], "--bench") == 0) { bench = true; noVsync = true; }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--no-vsync] [--bench]\n", argv[0]);
            std::printf("  --bench   run the per-stage benchmark (res 50/100/150/200) and exit\n");
            std::printf("Controls: LMB=move obstacle, SPACE/P=pause, G=grid, R=reset, Q/Esc=quit\n");
            return 0;
        }
    }

    std::printf("[flip-cuda] starting (GPU sim, CUDA-GL interop render)\n");

    AppWindow w;
    if (!createWindow(w, "FLIP Fluid (CUDA)")) return 1;
    loadGLBuffers();

    if (noVsync) {
        typedef int (*PFNGLXSWAPINTERVAL)(int);
        auto pfn = (PFNGLXSWAPINTERVAL)glXGetProcAddressARB(
            (const GLubyte*)"glXSwapIntervalMESA");
        if (pfn) pfn(0);
    }

    setupScene();
    scene.paused = true;

    if (bench) {
        runBenchmark(w);
        destroyWindow(w);
        delete scene.fluid;
        return 0;
    }

    bool mouseDownPrev = false;
    float mouseSimX = 0.0f, mouseSimY = 0.0f;
    bool mouseDown = false;
    int  mousePxX = 0, mousePxY = 0;
    bool mousePressedEdge = false;
    bool mouseReleasedEdge = false;
    bool dragOwnedByUI = false;

    auto fpsT0 = std::chrono::steady_clock::now();
    int  fpsFrames = 0;
    double lastFps = 0.0;
    bool gravityOn = (scene.gravity != 0.0f);

    while (w.running) {
        auto frameStart = std::chrono::steady_clock::now();
        mousePressedEdge = false;
        mouseReleasedEdge = false;

        while (XPending(w.dpy) > 0) {
            XEvent e;
            XNextEvent(w.dpy, &e);
            if (e.type == ClientMessage) {
                if ((Atom)e.xclient.data.l[0] == w.wm_delete) w.running = false;
            } else if (e.type == ConfigureNotify) {
                w.width  = e.xconfigure.width;
                w.height = e.xconfigure.height;
            } else if (e.type == KeyPress) {
                KeySym ks = XLookupKeysym(&e.xkey, 0);
                if (ks == XK_space || ks == XK_p || ks == XK_P) scene.paused = !scene.paused;
                else if (ks == XK_g || ks == XK_G) scene.showGrid = !scene.showGrid;
                else if (ks == XK_r || ks == XK_R) setupScene();
                else if (ks == XK_q || ks == XK_Q || ks == XK_Escape) w.running = false;
            } else if (e.type == ButtonPress && e.xbutton.button == Button1) {
                mouseDown = true; mousePressedEdge = true;
                mousePxX = e.xbutton.x; mousePxY = e.xbutton.y;
                mouseSimX = float(e.xbutton.x) / w.width  * simWidth;
                mouseSimY = (1.0f - float(e.xbutton.y) / w.height) * simHeight;
            } else if (e.type == ButtonRelease && e.xbutton.button == Button1) {
                mouseDown = false; mouseReleasedEdge = true;
                mousePxX = e.xbutton.x; mousePxY = e.xbutton.y;
                mouseSimX = float(e.xbutton.x) / w.width  * simWidth;
                mouseSimY = (1.0f - float(e.xbutton.y) / w.height) * simHeight;
            } else if (e.type == MotionNotify) {
                mousePxX = e.xmotion.x; mousePxY = e.xmotion.y;
                mouseSimX = float(e.xmotion.x) / w.width  * simWidth;
                mouseSimY = (1.0f - float(e.xmotion.y) / w.height) * simHeight;
            }
        }

        const int kPanelX = 10, kPanelY = 10, kPanelW = 160, kPanelH = 250;
        bool mouseOnPanel =
            (mousePxX >= kPanelX && mousePxX < kPanelX + kPanelW &&
             mousePxY >= kPanelY && mousePxY < kPanelY + kPanelH);
        if (mousePressedEdge && mouseOnPanel) dragOwnedByUI = true;
        if (!mouseDown) dragOwnedByUI = false;

        FlipFluidCuda& f = *scene.fluid;

        if (mouseDown && !dragOwnedByUI) {
            if (!mouseDownPrev) { setObstacle(mouseSimX, mouseSimY, true); scene.paused = false; }
            else                  setObstacle(mouseSimX, mouseSimY, false);
            mouseDownPrev = true;
        } else {
            if (mouseDownPrev) { scene.obstacleVelX = 0.0f; scene.obstacleVelY = 0.0f; }
            mouseDownPrev = false;
        }

        if (!scene.paused) {
            f.simulate(scene.dt, scene.gravity, scene.flipRatio,
                       scene.numPressureIters, scene.numParticleIters,
                       scene.overRelaxation, scene.compensateDrift,
                       scene.separateParticles,
                       scene.obstacleX, scene.obstacleY, scene.obstacleRadius,
                       scene.obstacleVelX, scene.obstacleVelY,
                       scene.numSubSteps);
            scene.frameNr += 1;
        }

        // ----- draw (T9_render) -----
        auto renderStart = std::chrono::steady_clock::now();
        glClear(GL_COLOR_BUFFER_BIT);
        setProjection(w.width, w.height);

        if (scene.showGrid)      drawGrid(f);
        if (scene.showParticles) drawParticles(f, w.height);
        if (scene.showObstacle)  drawObstacle(f, scene.obstacleX, scene.obstacleY,
                                              scene.obstacleRadius);

        flipcpu_ui::setProjectionToPixels(w.width, w.height);
        flipcpu_ui::Input uin;
        uin.screenW = w.width; uin.screenH = w.height;
        uin.mouseX  = mousePxX; uin.mouseY = mousePxY;
        uin.mouseDown    = mouseDown;
        uin.mousePressed = mousePressedEdge && mouseOnPanel;
        uin.mouseReleased = mouseReleasedEdge;
        flipcpu_ui::begin(uin);

        flipcpu_ui::beginPanel(kPanelX, kPanelY, kPanelW, kPanelH, "Controls");
        flipcpu_ui::text("FPS: %.1f", lastFps);
        flipcpu_ui::text("Render: %s", f.usingInterop() ? "INTEROP (B1)" : "D2H copy");
        flipcpu_ui::text("Particles: %d", f.numParticles);
        flipcpu_ui::text("Frame: %ld", scene.frameNr);
        flipcpu_ui::checkbox("Particles",          &scene.showParticles);
        flipcpu_ui::checkbox("Grid",               &scene.showGrid);
        flipcpu_ui::checkbox("Compensate Drift",   &scene.compensateDrift);
        flipcpu_ui::checkbox("Separate Particles", &scene.separateParticles);
        if (flipcpu_ui::checkbox("Gravity", &gravityOn))
            scene.gravity = gravityOn ? -9.81f : 0.0f;
        flipcpu_ui::slider("PIC <-> FLIP", &scene.flipRatio, 0.0f, 1.0f);
        float resFloat = (float)scene.resolution;
        flipcpu_ui::slider("Grid Res", &resFloat, 30.0f, 200.0f);
        int newRes = (int)(resFloat + 0.5f);
        if (newRes != scene.resolution) { scene.resolution = newRes; setupScene(); }
        flipcpu_ui::checkbox("Pause", &scene.paused);
        if (flipcpu_ui::button("Reset")) setupScene();
        flipcpu_ui::endPanel();
        flipcpu_ui::restoreProjection();

        glXSwapBuffers(w.dpy, w.xwin);

        // ----- timing: T9_render, T_total, 60-frame report -----
        auto frameEnd = std::chrono::steady_clock::now();
        auto msSince = [](std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        // Re-fetch: setupScene() above (res change / Reset) may have replaced fluid.
        FlipFluidCuda& ft = *scene.fluid;
        fliptiming::stageAdd(ft.timing, fliptiming::T9_render, msSince(renderStart, frameEnd));
        fliptiming::stageAdd(ft.timing, fliptiming::T_total,   msSince(frameStart, frameEnd));
        fliptiming::stageReportEvery(ft.timing, 60, "CUDA");

        fpsFrames += 1;
        double elapsed = std::chrono::duration<double>(frameEnd - fpsT0).count();
        if (elapsed >= 0.5) { lastFps = fpsFrames / elapsed; fpsT0 = frameEnd; fpsFrames = 0; }
    }

    destroyWindow(w);
    delete scene.fluid;
    return 0;
}
