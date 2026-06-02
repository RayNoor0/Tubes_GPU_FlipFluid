# CUDA build for the FLIP fluid demo (GPU sim + CUDA-GL interop render).
# Requires the CUDA toolkit (nvcc) and an NVIDIA GPU. Build/run from WSL2 (or Linux)
# with a working X server: `make && ./flip_cuda`.

NVCC      ?= nvcc
# -arch=native targets the GPU in this machine (CUDA >= 11.5). Override for older
# toolkits, e.g.  make ARCH="-arch=sm_75"
ARCH      ?= -arch=native
NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math $(ARCH)

CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O3 -Wall

LIBS := -lGL -lGLX -lX11 -lm

CU_SRC := main_cuda.cu flip_fluid_cuda.cu
CU_OBJ := $(CU_SRC:.cu=.o)
UI_OBJ := ui_host.o
BIN    := flip_cuda

all: $(BIN)

$(BIN): $(CU_OBJ) $(UI_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

%.o: %.cu flip_fluid_cuda.cuh ../stage_timing.h ../ui.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# The immediate-mode UI is plain host C++; compile it with the host compiler.
$(UI_OBJ): ../ui.cpp ../ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ ../ui.cpp

clean:
	rm -f $(CU_OBJ) $(UI_OBJ) $(BIN)

.PHONY: all clean
