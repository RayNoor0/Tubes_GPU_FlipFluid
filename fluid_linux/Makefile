CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -ffast-math -Wall -Wextra
LDFLAGS  ?=
LIBS     := -lGL -lGLX -lX11 -lm

SRC := flip_fluid.cpp ui.cpp main.cpp
OBJ := $(SRC:.cpp=.o)
BIN := flip

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp flip_fluid.h ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean
