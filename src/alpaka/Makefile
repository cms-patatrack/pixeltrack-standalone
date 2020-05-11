TARGETS = alpaka 
BUILD   = build
DEBUG   = build/debug

.PHONY: all debug clean $(TARGETS)

# general rules and targets
all: $(TARGETS)

debug: $(TARGETS:%=%-debug)

clean:
	rm -r -f test-* debug-* $(BUILD)  $(DEBUG) env.sh

$(BUILD):
	mkdir -p $(BUILD)

$(DEBUG):
	mkdir -p $(DEBUG)

# configure external tool here
BOOST_BASE  := /cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/boost/1.67.0-pafccj
TBB_BASE    :=
CUDA_BASE   := /usr/local/cuda-10.2.89
ALPAKA_BASE := ../../external/alpaka

# host compiler
CXX := g++
CXX_FLAGS := -O2 -std=c++14
CXX_DEBUG := -g
OMP_CXX_FLAGS := -fopenmp -foffload=disable
OMP_LD_FLAGS  := -fopenmp -foffload=disable

# CUDA compiler
ifdef CUDA_BASE
NVCC := $(CUDA_BASE)/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS := -O2 -std=c++14 --expt-relaxed-constexpr -w --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_60,code=sm_60 --generate-code arch=compute_70,code=sm_70
NVCC_DEBUG := -g -lineinfo

# CUDA flags for the host linker
CUDA_LIBDIR   := $(CUDA_BASE)/lib64
CUDA_LD_FLAGS := -L$(CUDA_LIBDIR) -lcudart -lcuda
endif

# boost flags
ifdef BOOST_BASE
BOOST_CXX_FLAGS := -I$(BOOST_BASE)/include
else
BOOST_CXX_FLAGS :=
endif

# TBB flags
ifdef TBB_BASE
TBB_LIBDIR    := $(TBB_BASE)/lib
TBB_CXX_FLAGS := -I$(TBB_BASE)/include
TBB_LD_FLAGS  := -L$(TBB_LIBDIR) -ltbb -lrt
else
TBB_LIBDIR    :=
TBB_CXX_FLAGS :=
TBB_LD_FLAGS  := -ltbb -lrt
endif

# alpaka flags
ifdef ALPAKA_BASE
ALPAKA_CXX_FLAGS  := $(BOOST_CXX_FLAGS) -I$(ALPAKA_BASE)/include
ALPAKA_NVCC_FLAGS := $(BOOST_CXX_FLAGS) -I$(ALPAKA_BASE)/include -Xcompiler
ALPAKA_LD_FLAGS   := 
ALPAKA_DEBUG      := -DALPAKA_DEBUG=1
endif


# color highlights for ANSI terminals
GREEN  := '\033[32m'
RED    := '\033[31m'
YELLOW := '\033[38;5;220m'
WHITE  := '\033[97m'
RESET  := '\033[0m'

# force the recreation of the environment file any time the Makefile is updated, before building any other target
-include environment
.PHONY: environment

environment: env.sh

env.sh: Makefile
	@echo '#! /bin/bash' > $@
ifdef ONEAPI_LIBDIR
	@echo 'export PATH=$(ONEAPI_BASE)/bin:$$PATH' >> $@
endif
	@echo -n 'export LD_LIBRARY_PATH=' >> $@
ifdef TBB_LIBDIR
	@echo -n '$(TBB_LIBDIR):' >> $@
endif
ifdef CUDA_LIBDIR
	@echo -n '$(CUDA_LIBDIR):' >> $@
endif
	@echo '$$LD_LIBRARY_PATH' >> $@
	@echo -e $(GREEN)Environment file$(RESET) regenerated, load the new envirnment with
	@echo
	@echo -e \ \ $(WHITE)source env.sh$(RESET)
	@echo


ifdef ALPAKA_BASE
alpaka: test-alpaka-serial test-alpaka-cuda #test-alpaka-tbb test-alpaka-omp2 test-alpaka-omp4 test-alpaka
	@echo -e $(GREEN)Alpaka targets built$(RESET)

alpaka-debug: debug-alpaka-serial debug-alpaka-tbb debug-alpaka-omp2 debug-alpaka-omp4 debug-alpaka-cuda debug-alpaka
	@echo -e $(GREEN)Alpaka debug targets built$(RESET)

# Alpaka implementation, compiled for the CPU serial backend
# $(BUILD)/rawtodigi_alpaka.serial.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) -o $@ -c $<

# $(BUILD)/analyzer_alpaka.serial.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_alpaka.serial.o: HeterogeneousCore/AlpakaUtilities/test/prefixScan_t.cc HeterogeneousCore/AlpakaUtilities/interface/prefixScan.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(ALPAKA_CXX_FLAGS) -o $@ -c $<

test-alpaka-serial: $(BUILD)/main_alpaka.serial.o #$(BUILD)/analyzer_alpaka.serial.o $(BUILD)/rawtodigi_alpaka.serial.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(ALPAKA_LD_FLAGS)

# $(DEBUG)/rawtodigi_alpaka.serial.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

# $(DEBUG)/analyzer_alpaka.serial.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

# $(DEBUG)/main_alpaka.serial.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

# debug-alpaka-serial: $(DEBUG)/main_alpaka.serial.o $(DEBUG)/analyzer_alpaka.serial.o $(DEBUG)/rawtodigi_alpaka.serial.o
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(ALPAKA_LD_FLAGS)

# # Alpaka implementation, compiled for the CPU parallel TBB backend
# $(BUILD)/rawtodigi_alpaka.tbb.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

# $(BUILD)/analyzer_alpaka.tbb.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

# $(BUILD)/main_alpaka.tbb.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

# test-alpaka-tbb: $(BUILD)/main_alpaka.tbb.o $(BUILD)/analyzer_alpaka.tbb.o $(BUILD)/rawtodigi_alpaka.tbb.o
# 	$(CXX) $(CXX_FLAGS) -o $@ $+ $(TBB_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# $(DEBUG)/rawtodigi_alpaka.tbb.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

# $(DEBUG)/analyzer_alpaka.tbb.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

# $(DEBUG)/main_alpaka.tbb.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

# debug-alpaka-tbb: $(DEBUG)/main_alpaka.tbb.o $(DEBUG)/analyzer_alpaka.tbb.o $(DEBUG)/rawtodigi_alpaka.tbb.o
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(TBB_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# # Alpaka implementation, compiled for the CPU parallel OpenMP 2.0 backend
# $(BUILD)/rawtodigi_alpaka.omp2.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

# $(BUILD)/analyzer_alpaka.omp2.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

# $(BUILD)/main_alpaka.omp2.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

# test-alpaka-omp2: $(BUILD)/main_alpaka.omp2.o $(BUILD)/analyzer_alpaka.omp2.o $(BUILD)/rawtodigi_alpaka.omp2.o
# 	$(CXX) $(CXX_FLAGS) -o $@ $+ $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# $(DEBUG)/rawtodigi_alpaka.omp2.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

# $(DEBUG)/analyzer_alpaka.omp2.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

# $(DEBUG)/main_alpaka.omp2.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

# debug-alpaka-omp2: $(DEBUG)/main_alpaka.omp2.o $(DEBUG)/analyzer_alpaka.omp2.o $(DEBUG)/rawtodigi_alpaka.omp2.o
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# # Alpaka implementation, compiled for the CPU parallel OpenMP 4.0 backend
# $(BUILD)/rawtodigi_alpaka.omp4.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

# $(BUILD)/analyzer_alpaka.omp4.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

# $(BUILD)/main_alpaka.omp4.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

# test-alpaka-omp4: $(BUILD)/main_alpaka.omp4.o $(BUILD)/analyzer_alpaka.omp4.o $(BUILD)/rawtodigi_alpaka.omp4.o
# 	$(CXX) $(CXX_FLAGS) -o $@ $+ $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# $(DEBUG)/rawtodigi_alpaka.omp4.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

# $(DEBUG)/analyzer_alpaka.omp4.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

# $(DEBUG)/main_alpaka.omp4.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

# debug-alpaka-omp4: $(DEBUG)/main_alpaka.omp4.o $(DEBUG)/analyzer_alpaka.omp4.o $(DEBUG)/rawtodigi_alpaka.omp4.o
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

ifdef CUDA_BASE
# # Alpaka implementation, compiled for the GPU CUDA backend
# $(BUILD)/rawtodigi_alpaka.cuda.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(NVCC) $(NVCC_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_NVCC_FLAGS) -o $@ -x cu -dc $<

# $(BUILD)/analyzer_alpaka.cuda.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
# 	$(NVCC) $(NVCC_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_NVCC_FLAGS) -o $@ -x cu -dc $<

# $(BUILD)/alpaka.dlink.o: $(BUILD)/rawtodigi_alpaka.cuda.o $(BUILD)/analyzer_alpaka.cuda.o
# 	$(NVCC) $(NVCC_FLAGS) -o $@ -dlink $^

$(BUILD)/main_alpaka.cuda.o: HeterogeneousCore/AlpakaUtilities/test/prefixScan_t.cc HeterogeneousCore/AlpakaUtilities/interface/prefixScan.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) -o $@  -x cu -dc $<

test-alpaka-cuda: $(BUILD)/main_alpaka.cuda.o
	$(NVCC) $(NVCC_FLAGS) $(ALPAKA_CXX_FLAGS) -o $@ $+ $(CUDA_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# $(DEBUG)/rawtodigi_alpaka.cuda.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_NVCC_FLAGS) $(ALPAKA_DEBUG) -o $@ -x cu -dc $<

# $(DEBUG)/analyzer_alpaka.cuda.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h loops.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
# 	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_NVCC_FLAGS) $(ALPAKA_DEBUG) -o $@ -x cu -dc $<

# $(DEBUG)/alpaka.dlink.o: $(DEBUG)/rawtodigi_alpaka.cuda.o $(DEBUG)/analyzer_alpaka.cuda.o
# 	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -o $@ -dlink $^

# $(DEBUG)/main_alpaka.cuda.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

# debug-alpaka-cuda: $(DEBUG)/main_alpaka.cuda.o $(DEBUG)/analyzer_alpaka.cuda.o $(DEBUG)/rawtodigi_alpaka.cuda.o $(DEBUG)/alpaka.dlink.o
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ $+ $(CUDA_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# # Alpaka implementation with run-time device choice
# $(BUILD)/main_alpaka.o: main_alpaka.cc analyzer_alpaka.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) -o $@ -c $<

# test-alpaka: $(BUILD)/main_alpaka.o $(BUILD)/rawtodigi_alpaka.serial.o $(BUILD)/rawtodigi_alpaka.tbb.o $(BUILD)/rawtodigi_alpaka.omp2.o $(BUILD)/rawtodigi_alpaka.omp4.o $(BUILD)/rawtodigi_alpaka.cuda.o $(BUILD)/analyzer_alpaka.serial.o $(BUILD)/analyzer_alpaka.tbb.o $(BUILD)/analyzer_alpaka.omp2.o $(BUILD)/analyzer_alpaka.omp4.o $(BUILD)/analyzer_alpaka.cuda.o $(BUILD)/alpaka.dlink.o
# 	$(CXX) $(CXX_FLAGS) -o $@ $+ $(CUDA_LD_FLAGS) $(TBB_LD_FLAGS) $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# $(DEBUG)/main_alpaka.o: main_alpaka.cc analyzer_alpaka.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

# debug-alpaka: $(DEBUG)/main_alpaka.o $(DEBUG)/rawtodigi_alpaka.serial.o $(DEBUG)/rawtodigi_alpaka.tbb.o $(DEBUG)/rawtodigi_alpaka.omp2.o $(DEBUG)/rawtodigi_alpaka.omp4.o $(DEBUG)/rawtodigi_alpaka.cuda.o $(DEBUG)/analyzer_alpaka.serial.o $(DEBUG)/analyzer_alpaka.tbb.o $(DEBUG)/analyzer_alpaka.omp2.o $(DEBUG)/analyzer_alpaka.omp4.o $(DEBUG)/analyzer_alpaka.cuda.o $(DEBUG)/alpaka.dlink.o
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(CUDA_LD_FLAGS) $(TBB_LD_FLAGS) $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# else
# test-alpaka-cuda:
# 	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Alpaka targets using CUDA will not be built

# debug-alpaka-cuda:
# 	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Alpaka debug targets using CUDA will not be built

# # Alpaka implementation with run-time device choice, with the CUDA backend disabled
# $(BUILD)/main_alpaka.o: main_alpaka.cc analyzer_alpaka.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
# 	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) -o $@ -c $<

# test-alpaka: $(BUILD)/main_alpaka.o $(BUILD)/rawtodigi_alpaka.serial.o $(BUILD)/rawtodigi_alpaka.tbb.o $(BUILD)/rawtodigi_alpaka.omp2.o $(BUILD)/rawtodigi_alpaka.omp4.o $(BUILD)/analyzer_alpaka.serial.o $(BUILD)/analyzer_alpaka.tbb.o $(BUILD)/analyzer_alpaka.omp2.o $(BUILD)/analyzer_alpaka.omp4.o
# 	$(CXX) $(CXX_FLAGS) -o $@ $+ $(TBB_LD_FLAGS) $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# $(DEBUG)/main_alpaka.o: main_alpaka.cc analyzer_alpaka.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

# debug-alpaka: $(DEBUG)/main_alpaka.o $(DEBUG)/rawtodigi_alpaka.serial.o $(DEBUG)/rawtodigi_alpaka.tbb.o $(DEBUG)/rawtodigi_alpaka.omp2.o $(DEBUG)/rawtodigi_alpaka.omp4.o $(DEBUG)/analyzer_alpaka.serial.o $(DEBUG)/analyzer_alpaka.tbb.o $(DEBUG)/analyzer_alpaka.omp2.o $(DEBUG)/analyzer_alpaka.omp4.o
# 	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(TBB_LD_FLAGS) $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

endif

# else
# alpaka:
# 	@echo -e $(YELLOW)Alpaka not found$(RESET), Alpaka targets will not be built

endif


 

