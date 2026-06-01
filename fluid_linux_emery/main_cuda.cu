// Window + renderer + main loop for the CUDA FLIP demo.
// Simulation runs on the GPU (flipgpu::FlipFluidCuda); a host flipcpu::FlipFluid
// is kept purely as the scene-setup + render-staging object. Each frame the
// render-relevant arrays are copied D2H (no GL interop in this build) and drawn
// with the same legacy OpenGL path as the CPU demo.
//
// Inputs:
//   left mouse drag   move/release the obstacle
//   SPACE / P         pause-resume
//   G                 toggle grid
//   R                 reset scene
//   Q / Esc           quit

#include "flip_fluid.h"
#include "flip_fluid_cuda.h"
#include "ui.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using flipcpu::FlipFluid;
using flipgpu::FlipFluidCuda;

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
    FlipFluid*     fluid  = nullptr;   // host staging (scene setup + render)
    FlipFluidCuda* gpu    = nullptr;   // device simulation
    flipgpu::SolverType solver = flipgpu::SOLVER_RED_BLACK;
};

static Scene scene;

constexpr int CANVAS_W = 900;
constexpr int CANVAS_H = 700;
constexpr float simHeight = 3.0f;
constexpr float cScale = float(CANVAS_H) / simHeight;
constexpr float simWidth = float(CANVAS_W) / cScale;

// --------------------------- scene helpers ----------------------------------
static void carveObstacle(FlipFluid& f, float x, float y, float r,
                          float vx, float vy) {
    int n = f.fNumY;
    for (int i = 1; i < f.fNumX - 2; ++i) {
        for (int j = 1; j < f.fNumY - 2; ++j) {
            f.s[i * n + j] = 1.0f;
            float dx = (i + 0.5f) * f.h - x;
            float dy = (j + 0.5f) * f.h - y;
            if (dx * dx + dy * dy < r * r) {
                f.s[i * n + j] = 0.0f;
                f.u[i * n + j]       = vx;
                f.u[(i + 1) * n + j] = vx;
                f.v[i * n + j]       = vy;
                f.v[i * n + j + 1]   = vy;
            }
        }
    }
}

static void setObstacle(float x, float y, bool reset) {
    float vx = 0.0f, vy = 0.0f;
    if (!reset) {
        vx = (x - scene.obstacleX) / scene.dt;
        vy = (y - scene.obstacleY) / scene.dt;
    }
    scene.obstacleX = x;
    scene.obstacleY = y;
    carveObstacle(*scene.fluid, x, y, scene.obstacleRadius, vx, vy);
    if (scene.gpu) scene.gpu->uploadSolid(*scene.fluid);   // refresh device mask
    scene.showObstacle  = true;
    scene.obstacleVelX  = vx;
    scene.obstacleVelY  = vy;
}

static void seedParticles(FlipFluid& f, int numX, int numY,
                          float h, float r, float dx, float dy) {
    for (int i = 0; i < numX; ++i) {
        for (int j = 0; j < numY; ++j) {
            int pid = i * numY + j;
            float offset = (j % 2 == 0) ? 0.0f : r;
            f.particlePosX[pid] = h + r + dx * i + offset;
            f.particlePosY[pid] = h + r + dy * j;
        }
    }
}

static void setupTank(FlipFluid& f) {
    int n = f.fNumY;
    for (int i = 0; i < f.fNumX; ++i) {
        for (int j = 0; j < f.fNumY; ++j) {
            float sVal = 1.0f;
            if (i == 0 || i == f.fNumX - 1 || j == 0) sVal = 0.0f;
            f.s[i * n + j] = sVal;
        }
    }
}

static void setupScene() {
    scene.obstacleRadius   = 0.15f;
    scene.overRelaxation   = 1.9f;
    scene.dt               = 1.0f / 60.0f;
    scene.numParticleIters = 2;

    int res = scene.resolution;
    if      (res <= 100) scene.numSubSteps = 1;
    else if (res <= 140) scene.numSubSteps = 2;
    else if (res <= 180) scene.numSubSteps = 3;
    else                 scene.numSubSteps = 4;
    scene.numPressureIters = 50 + std::max(0, (res - 100)) / 2;

    float tankHeight  = 1.0f * simHeight;
    float tankWidth   = 1.0f * simWidth;
    float h           = tankHeight / res;
    float density     = 1000.0f;

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

    delete scene.gpu;   scene.gpu = nullptr;
    delete scene.fluid;
    scene.fluid = new FlipFluid(density, tankWidth, tankHeight, h, r, maxParticles);

    FlipFluid& f = *scene.fluid;
    f.numParticles = numX * numY;
    f.particleRestDensity = 0.0f;

    seedParticles(f, numX, numY, h, r, dx, dy);
    setupTank(f);
    setObstacle(3.0f, 2.0f, true);          // static obstacle (gpu not yet built)

    // Build the device simulator and upload the freshly-seeded state.
    scene.gpu = new FlipFluidCuda(density, tankWidth, tankHeight, h, r, maxParticles);
    scene.gpu->solver = scene.solver;
    scene.gpu->uploadStateFrom(f);

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

static void drawGrid(const FlipFluid& f) {
    float h = f.h;
    glBegin(GL_QUADS);
    for (int i = 0; i < f.fNumX; ++i) {
        for (int j = 0; j < f.fNumY; ++j) {
            int idx = i * f.fNumY + j;
            float r = f.cellColor[3 * idx + 0];
            float g = f.cellColor[3 * idx + 1];
            float b = f.cellColor[3 * idx + 2];
            float x0 = i * h, y0 = j * h;
            float x1 = x0 + h, y1 = y0 + h;
            glColor3f(r, g, b);
            glVertex2f(x0, y0);
            glVertex2f(x1, y0);
            glVertex2f(x1, y1);
            glVertex2f(x0, y1);
        }
    }
    glEnd();
}

static void drawParticles(const FlipFluid& f, int viewportH) {
    float pxPerSimUnit = float(viewportH) / simHeight;
    float diameterPx = 2.0f * f.particleRadius * pxPerSimUnit;
    if (diameterPx < 1.0f) diameterPx = 1.0f;
    glPointSize(diameterPx);
    glBegin(GL_POINTS);
    for (int i = 0; i < f.numParticles; ++i) {
        glColor3f(f.particleColorR[i], f.particleColorG[i], f.particleColorB[i]);
        glVertex2f(f.particlePosX[i], f.particlePosY[i]);
    }
    glEnd();
}

static void drawObstacle(const FlipFluid& f, float ox, float oy, float orad) {
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

static void renderSimFrame(AppWindow& w, const FlipFluid& f) {
    glClear(GL_COLOR_BUFFER_BIT);
    setProjection(w.width, w.height);
    if (scene.showGrid)      drawGrid(f);
    if (scene.showParticles) drawParticles(f, w.height);
    if (scene.showObstacle)  drawObstacle(f, scene.obstacleX, scene.obstacleY,
                                          scene.obstacleRadius);
}

// --------------------------- benchmark / validation -------------------------
static const int kBenchResolutions[] = {50, 100, 150, 200};

struct SimStats { double meanSpeed, totalKE, centroidX, centroidY; };

static SimStats computeStats(const FlipFluid& f) {
    double sumSpeed = 0.0, sumKE = 0.0, cx = 0.0, cy = 0.0;
    for (int i = 0; i < f.numParticles; ++i) {
        double vx = f.particleVelX[i], vy = f.particleVelY[i];
        double v2 = vx * vx + vy * vy;
        sumSpeed += std::sqrt(v2);
        sumKE    += 0.5 * v2;
        cx += f.particlePosX[i];
        cy += f.particlePosY[i];
    }
    int n = std::max(1, f.numParticles);
    return { sumSpeed / n, sumKE, cx / n, cy / n };
}

static void stepFixed() {
    scene.gpu->simulate(scene.dt, scene.gravity, scene.flipRatio,
                        scene.numPressureIters, scene.numParticleIters,
                        scene.overRelaxation, scene.compensateDrift,
                        scene.separateParticles,
                        scene.obstacleX, scene.obstacleY, scene.obstacleRadius,
                        0.0f, 0.0f, scene.numSubSteps);
}

static const char* solverName() {
    return scene.solver == flipgpu::SOLVER_JACOBI ? "cuda-jacobi" : "cuda-rb";
}

static void runBenchmark(AppWindow& w, int singleRes) {
    const int warmup = 60, measure = 600;
    std::FILE* csv = std::fopen("bench_cuda.csv", "w");
    if (csv) flipbench::StageTimers::writeCsvHeader(csv);

    int resList[4]; int nRes;
    if (singleRes > 0) { resList[0] = singleRes; nRes = 1; }
    else { for (int i = 0; i < 4; ++i) resList[i] = kBenchResolutions[i]; nRes = 4; }

    for (int ri = 0; ri < nRes; ++ri) {
        scene.resolution = resList[ri];
        setupScene();
        scene.paused = false;
        FlipFluid& host = *scene.fluid;

        flipbench::StageTimers timers;
        timers.resolution = scene.resolution;
        scene.gpu->timers = &timers;

        std::printf("[flip-cuda] benchmarking res=%d particles=%d pIters=%d solver=%s ...\n",
                    scene.resolution, host.numParticles, scene.numPressureIters, solverName());

        for (int frame = 0; frame < warmup + measure; ++frame) {
            timers.measuring = (frame >= warmup);
            auto t0 = std::chrono::steady_clock::now();

            stepFixed();
            scene.gpu->downloadForRender(host);     // T10_d2h (timed inside)

            auto r0 = std::chrono::steady_clock::now();
            renderSimFrame(w, host);
            glXSwapBuffers(w.dpy, w.xwin);
            auto r1 = std::chrono::steady_clock::now();
            timers.add(flipbench::T9_render,
                std::chrono::duration<double, std::milli>(r1 - r0).count());

            auto t1 = std::chrono::steady_clock::now();
            timers.frameTotal = std::chrono::duration<double, std::milli>(t1 - t0).count();
            timers.endFrame();

            while (XPending(w.dpy) > 0) { XEvent e; XNextEvent(w.dpy, &e); }
        }

        timers.numParticles = host.numParticles;
        timers.printSummary(solverName());
        if (csv) timers.writeCsvRow(csv, solverName());
        scene.gpu->timers = nullptr;
    }
    if (csv) std::fclose(csv);
    std::printf("[flip-cuda] benchmark complete -> bench_cuda.csv\n");
}

static void runDumpStats(int singleRes, int frames) {
    scene.resolution = (singleRes > 0) ? singleRes : 100;
    setupScene();
    scene.paused = false;
    FlipFluid& host = *scene.fluid;

    std::FILE* csv = std::fopen("stats_cuda.csv", "w");
    if (csv) std::fprintf(csv, "frame,meanSpeed,totalKE,centroidX,centroidY\n");
    for (int frame = 0; frame < frames; ++frame) {
        stepFixed();
        scene.gpu->downloadParticles(host);
        SimStats s = computeStats(host);
        if (csv) std::fprintf(csv, "%d,%.8f,%.8f,%.8f,%.8f\n",
                              frame, s.meanSpeed, s.totalKE, s.centroidX, s.centroidY);
    }
    if (csv) std::fclose(csv);
    std::printf("[flip-cuda] dump-stats complete (res=%d, %d frames, solver=%s) -> stats_cuda.csv\n",
                scene.resolution, frames, solverName());
}

// --------------------------- main -------------------------------------------
int main(int argc, char** argv) {
    bool noVsync   = false;
    bool benchmark = false;
    bool dumpStats = false;
    int  argRes    = 0;
    int  statFrames = 300;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-vsync") == 0) noVsync = true;
        else if (std::strcmp(argv[i], "--benchmark") == 0) benchmark = true;
        else if (std::strcmp(argv[i], "--dump-stats") == 0) dumpStats = true;
        else if (std::strcmp(argv[i], "--res") == 0 && i + 1 < argc) argRes = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) statFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--solver") == 0 && i + 1 < argc) {
            ++i;
            if (std::strcmp(argv[i], "jacobi") == 0) scene.solver = flipgpu::SOLVER_JACOBI;
            else scene.solver = flipgpu::SOLVER_RED_BLACK;
        }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--no-vsync] [--benchmark] [--dump-stats] [--res N] [--frames N] [--solver rb|jacobi]\n", argv[0]);
            std::printf("  --benchmark        warmup(60)+measure(600) per-stage timing sweep -> bench_cuda.csv\n");
            std::printf("  --dump-stats       per-frame aggregate scalars -> stats_cuda.csv\n");
            std::printf("  --res N            fix grid resolution N (default: sweep 50/100/150/200 in benchmark)\n");
            std::printf("  --solver rb|jacobi pressure solver (default rb = Red-Black Gauss-Seidel)\n");
            std::printf("Controls: LMB=move obstacle, SPACE/P=pause, G=grid, R=reset, Q/Esc=quit\n");
            return 0;
        }
    }

    std::printf("[flip-cuda] starting (GPU sim, GPU render), solver=%s\n", solverName());
    setupScene();
    scene.paused = true;

    AppWindow w;
    if (!createWindow(w, "FLIP Fluid (CUDA GPU sim)")) return 1;

    if (noVsync) {
        typedef int (*PFNGLXSWAPINTERVAL)(int);
        auto pfn = (PFNGLXSWAPINTERVAL)glXGetProcAddressARB(
            (const GLubyte*)"glXSwapIntervalMESA");
        if (pfn) pfn(0);
    }

    if (benchmark) {
        runBenchmark(w, argRes);
        destroyWindow(w);
        delete scene.gpu; delete scene.fluid;
        return 0;
    }
    if (dumpStats) {
        runDumpStats(argRes, statFrames);
        destroyWindow(w);
        delete scene.gpu; delete scene.fluid;
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

        FlipFluid& host = *scene.fluid;

        if (mouseDown && !dragOwnedByUI) {
            if (!mouseDownPrev) { setObstacle(mouseSimX, mouseSimY, true); scene.paused = false; }
            else                  setObstacle(mouseSimX, mouseSimY, false);
            mouseDownPrev = true;
        } else {
            if (mouseDownPrev) { scene.obstacleVelX = 0.0f; scene.obstacleVelY = 0.0f; }
            mouseDownPrev = false;
        }

        if (!scene.paused) {
            scene.gpu->simulate(scene.dt, scene.gravity, scene.flipRatio,
                                scene.numPressureIters, scene.numParticleIters,
                                scene.overRelaxation, scene.compensateDrift,
                                scene.separateParticles,
                                scene.obstacleX, scene.obstacleY, scene.obstacleRadius,
                                scene.obstacleVelX, scene.obstacleVelY,
                                scene.numSubSteps);
            scene.gpu->downloadForRender(host);
            scene.frameNr += 1;
        }

        renderSimFrame(w, host);

        flipcpu_ui::setProjectionToPixels(w.width, w.height);
        flipcpu_ui::Input uin;
        uin.screenW = w.width; uin.screenH = w.height;
        uin.mouseX = mousePxX; uin.mouseY = mousePxY;
        uin.mouseDown = mouseDown;
        uin.mousePressed = mousePressedEdge && mouseOnPanel;
        uin.mouseReleased = mouseReleasedEdge;
        flipcpu_ui::begin(uin);

        flipcpu_ui::beginPanel(kPanelX, kPanelY, kPanelW, kPanelH, "Controls");
        flipcpu_ui::text("FPS: %.1f", lastFps);
        flipcpu_ui::text("Particles: %d", host.numParticles);
        flipcpu_ui::text("Frame: %ld", scene.frameNr);
        flipcpu_ui::text("Solver: %s", solverName());
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

        fpsFrames += 1;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fpsT0).count();
        if (elapsed >= 0.5) {
            lastFps = fpsFrames / elapsed;
            std::printf("[flip-cuda] %.1f FPS  particles=%d frame=%ld\n",
                        lastFps, host.numParticles, scene.frameNr);
            std::fflush(stdout);
            fpsT0 = now; fpsFrames = 0;
        }
    }

    destroyWindow(w);
    delete scene.gpu;
    delete scene.fluid;
    return 0;
}
