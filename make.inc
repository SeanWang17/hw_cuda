
CC      = cc
NVCC    = nvcc
CFLAGS  = -O3
NVFLAGS = -O3

CUDA = /Developer/NVIDIA/cuda
MKL  = /opt/intel/mkl

INC = -I$(CUDA)/include -I$(MKL)/include
LIB = -L$(CUDA)/lib -L$(MKL)/lib
LIBS = -lcudart -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
