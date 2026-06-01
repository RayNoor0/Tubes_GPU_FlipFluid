# CPU build (g++) -------------------------------------------------------------
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -ffast-math -Wall -Wextra
LDFLAGS  ?=
LIBS     := -lGL -lGLX -lX11 -lm

SRC := flip_fluid.cpp ui.cpp main.cpp
OBJ := $(SRC:.cpp=.o)
BIN := flip

# CUDA build (nvcc) ------------------------------------------------------------
# Override NVCC_ARCH for your GPU, e.g. sm_70 (V100), sm_75 (T4/RTX20xx),
# sm_80 (A100), sm_86 (RTX30xx), sm_89 (RTX40xx). Check with `nvidia-smi`/deviceQuery.
NVCC      ?= nvcc
NVCC_ARCH ?= sm_75
NVCCFLAGS ?= -std=c++17 -O3 -arch=$(NVCC_ARCH) -Xcompiler -O3
CUDA_SRC  := flip_fluid_cuda.cu main_cuda.cu ui.cpp flip_fluid.cpp
CUDA_LIBS := -lGL -lGLX -lX11
CUDA_BIN  := flip_cuda

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp flip_fluid.h ui.h bench.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Single nvcc invocation (avoids ui.o colliding with the CPU build's object).
cuda: $(CUDA_SRC) flip_fluid_cuda.h flip_fluid.h ui.h bench.h
	$(NVCC) $(NVCCFLAGS) -o $(CUDA_BIN) $(CUDA_SRC) $(CUDA_LIBS)

clean:
	rm -f $(OBJ) $(BIN) $(CUDA_BIN)

clean-csv:
	rm -f bench_cpu.csv bench_cuda.csv stats_cpu.csv stats_cuda.csv

.PHONY: all cuda clean clean-csv
