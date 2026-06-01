# CPU build (g++) -------------------------------------------------------------
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -ffast-math -Wall -Wextra
LDFLAGS  ?=
LIBS     := -lGL -lGLX -lX11 -lm

SRC := flip_fluid.cpp ui.cpp main.cpp
OBJ := $(SRC:.cpp=.o)
BIN := flip

# CUDA build (nvcc) ------------------------------------------------------------
# NVCC_ARCH is auto-detected from the server's GPU via nvidia-smi (compute_cap),
# e.g. 8.6 -> sm_86. Falls back to sm_75 if nvidia-smi is unavailable. You can
# still override it manually: `make cuda NVCC_ARCH=sm_80`.
NVCC      ?= nvcc
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
NVCC_ARCH ?= $(if $(DETECTED_CC),sm_$(DETECTED_CC),sm_75)
NVCCFLAGS ?= -std=c++17 -O3 -arch=$(NVCC_ARCH) -Xcompiler -O3
CUDA_SRC  := flip_fluid_cuda.cu main_cuda.cu ui.cpp flip_fluid.cpp
CUDA_LIBS := -lGL -lGLX -lX11
CUDA_BIN  := flip_cuda

# Run helpers ------------------------------------------------------------------
# On a headless server (no DISPLAY) the X11 window is opened inside a virtual
# framebuffer via xvfb-run; if DISPLAY is already set, run directly. Tunables:
#   SOLVER=rb|jacobi   RES=N   FRAMES=N
XVFB   ?= xvfb-run -a -s "-screen 0 1280x720x24"
RUN    := $(if $(DISPLAY),,$(XVFB))
SOLVER ?= rb
RES    ?= 100
FRAMES ?= 300

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp flip_fluid.h ui.h bench.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Single nvcc invocation (avoids ui.o colliding with the CPU build's object).
cuda: $(CUDA_SRC) flip_fluid_cuda.h flip_fluid.h ui.h bench.h
	$(NVCC) $(NVCCFLAGS) -o $(CUDA_BIN) $(CUDA_SRC) $(CUDA_LIBS)

# Convenience run targets ------------------------------------------------------
gpu-info:
	@echo "Detected compute_cap -> NVCC_ARCH = $(NVCC_ARCH)"
	@nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>/dev/null \
		|| echo "nvidia-smi not found (no GPU / driver?)"

# Interactive windows (needs a real display or xvfb)
run-cpu: $(BIN)
	$(RUN) ./$(BIN) --no-vsync
run-gpu: cuda
	$(RUN) ./$(CUDA_BIN) --no-vsync --solver $(SOLVER)

# Benchmark sweep -> bench_cpu.csv / bench_cuda.csv
bench-cpu: $(BIN)
	$(RUN) ./$(BIN) --benchmark --no-vsync
bench-gpu: cuda
	$(RUN) ./$(CUDA_BIN) --benchmark --no-vsync --solver $(SOLVER)
bench-gpu-jacobi: cuda
	$(RUN) ./$(CUDA_BIN) --benchmark --no-vsync --solver jacobi
bench-all: bench-cpu bench-gpu bench-gpu-jacobi

# Numerical validation -> stats_cpu.csv / stats_cuda.csv
stats-cpu: $(BIN)
	$(RUN) ./$(BIN) --dump-stats --res $(RES) --frames $(FRAMES)
stats-gpu: cuda
	$(RUN) ./$(CUDA_BIN) --dump-stats --res $(RES) --frames $(FRAMES) --solver $(SOLVER)
stats-all: stats-cpu stats-gpu

clean:
	rm -f $(OBJ) $(BIN) $(CUDA_BIN)

clean-csv:
	rm -f bench_cpu.csv bench_cuda.csv stats_cpu.csv stats_cuda.csv

.PHONY: all cuda gpu-info run-cpu run-gpu bench-cpu bench-gpu bench-gpu-jacobi \
        bench-all stats-cpu stats-gpu stats-all clean clean-csv
