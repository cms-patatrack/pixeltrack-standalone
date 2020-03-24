export BASE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Build flags
export CXX := g++
USER_CXXFLAGS :=
export CXXFLAGS := -O2 -std=c++1z -pthread -pipe -fPIC -ftree-vectorize -fvisibility-inlines-hidden -fno-math-errno --param vect-max-version-for-alias-checks=50 -Xassembler --compress-debug-sections -msse3 -felide-constructors -fmessage-length=0 -fdiagnostics-show-option -Werror=main -Werror=pointer-arith -Werror=overlength-strings -Wno-vla -Werror=overflow -Wstrict-overflow -Werror=array-bounds -Werror=format-contains-nul -Werror=type-limits -Wall -Wno-non-template-friend -Wno-long-long -Wreturn-type -Wunused -Wparentheses -Wno-deprecated -Werror=return-type -Werror=missing-braces -Werror=unused-value -Werror=address -Werror=format -Werror=sign-compare -Werror=write-strings -Werror=delete-non-virtual-dtor -Werror=strict-aliasing -Werror=narrowing -Werror=unused-but-set-variable -Werror=reorder -Werror=unused-variable -Werror=conversion-null -Werror=return-local-addr -Wnon-virtual-dtor -Werror=switch -Wno-unused-local-typedefs -Wno-attributes -Wno-psabi -Wno-error=unused-variable $(USER_CXXFLAGS)
export LDFLAGS := -pthread -Wl,-E -lstdc++fs
export SO_LDFLAGS := -Wl,-z,defs

CLANG_FORMAT := clang-format-8

# Source code
export SRC_DIR := $(BASE_DIR)/src

# Directory where to put object and dependency files
export OBJ_DIR := $(BASE_DIR)/obj

# Directory where to put libraries
export LIB_DIR := $(BASE_DIR)/lib

# System external definitions
CUDA_BASE := /usr/local/cuda
CUDA_LIBDIR := $(CUDA_BASE)/lib64
USER_CUDAFLAGS :=
export CUDA_DEPS := $(CUDA_BASE)/lib64/libcudart.so
export CUDA_CXXFLAGS := -I$(CUDA_BASE)/include
export CUDA_LDFLAGS := -L$(CUDA_BASE)/lib64 -lcudart -lcudadevrt
export CUDA_NVCC := $(CUDA_BASE)/bin/nvcc
CUDA_CUCOMMON := -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -O3 -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info --source-in-ptx --cudart=shared --compiler-options '-O2 -pthread -pipe -Werror=main -Werror=pointer-arith -Werror=overlength-strings -Wno-vla -Werror=overflow -ftree-vectorize -Wstrict-overflow -Werror=array-bounds -Werror=format-contains-nul -Werror=type-limits -fvisibility-inlines-hidden -fno-math-errno --param vect-max-version-for-alias-checks=50 -Xassembler --compress-debug-sections -msse3 -felide-constructors -fmessage-length=0 -Wall -Wno-non-template-friend -Wno-long-long -Wreturn-type -Wunused -Wparentheses -Wno-deprecated -Werror=return-type -Werror=missing-braces -Werror=unused-value -Werror=address -Werror=format -Werror=sign-compare -Werror=write-strings -Werror=delete-non-virtual-dtor -Werror=strict-aliasing -Werror=narrowing -Werror=unused-but-set-variable -Werror=reorder -Werror=unused-variable -Werror=conversion-null -Werror=return-local-addr -Wnon-virtual-dtor -Werror=switch -fdiagnostics-show-option -Wno-unused-local-typedefs -Wno-attributes -Wno-psabi -Ofast -Wno-error=unused-variable -Wno-error=unused-variable -Wno-error=unused-variable -std=c++14  -fPIC'
export CUDA_CUFLAGS := -dc $(CUDA_CUCOMMON) $(USER_CUDAFLAGS)
export CUDA_DLINKFLAGS := -dlink $(CUDA_CUCOMMON)

# Input data definitions
DATA_BASE := $(BASE_DIR)/data
export DATA_DEPS := $(DATA_BASE)/data_ok
DATA_TAR_GZ := $(DATA_BASE)/data.tar.gz

# External definitions
EXTERNAL_BASE := $(BASE_DIR)/external
TBB_BASE := $(EXTERNAL_BASE)/tbb
TBB_LIBDIR := $(TBB_BASE)/lib
TBB_LIB := $(TBB_LIBDIR)/libtbb.so
export TBB_DEPS := $(TBB_LIB)
export TBB_CXXFLAGS := -I$(TBB_BASE)/include
export TBB_LDFLAGS := -L$(TBB_LIBDIR) -ltbb

CUB_BASE := $(EXTERNAL_BASE)/cub
export CUB_DEPS := $(CUB_BASE)
export CUB_CXXFLAGS := -I$(CUB_BASE)
export CUB_LDFLAGS :=

EIGEN_BASE := $(EXTERNAL_BASE)/eigen
export EIGEN_DEPS := $(EIGEN_BASE)
export EIGEN_CXXFLAGS := -I$(EIGEN_BASE)
export EIGEN_LDFLAGS :=

KOKKOS_BASE := $(EXTERNAL_BASE)/kokkos
KOKKOS_SRC := $(KOKKOS_BASE)/source
KOKKOS_BUILD := $(KOKKOS_BASE)/build
KOKKOS_INSTALL := $(KOKKOS_BASE)/install
KOKKOS_LIB := $(KOKKOS_INSTALL)/lib/libkokkoscore.a
KOKKOS_MAKEFILE := $(KOKKOS_BUILD)/Makefile
KOKKOS_CMAKEFLAGS := -DCMAKE_INSTALL_PREFIX=$(KOKKOS_INSTALL) \
                     -DKokkos_CXX_STANDARD=14 \
                     -DCMAKE_CXX_COMPILER=$(KOKKOS_SRC)/bin/nvcc_wrapper -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=On -DKokkos_CUDA_DIR=$(CUDA_BASE) -DKokkos_ARCH_PASCAL60=On
# if without CUDA, replace the above line with
#                     -DCMAKE_CXX_COMPILER=g++
export KOKKOS_DEPS := $(KOKKOS_LIB)
export KOKKOS_CXXFLAGS := -I$(KOKKOS_INSTALL)/include
export KOKKOS_LDFLAGS := -L$(KOKKOS_INSTALL)/lib

# force the recreation of the environment file any time the Makefile is updated, before building any other target
-include environment

# Targets and their dependencies on externals
TARGETS := $(notdir $(wildcard $(SRC_DIR)/*))
all: $(TARGETS)
# $(TARGETS) needs to be PHONY because only the called Makefile knows their dependencies
.PHONY: $(TARGETS) all environment format clean distclean dataclean external_tbb external_cub external_eigen external_kokkos

environment: env.sh
env.sh: Makefile
	@echo '#! /bin/bash' > $@
	@echo -n 'export LD_LIBRARY_PATH=' >> $@
	@echo -n '$(TBB_LIBDIR):' >> $@
	@echo -n '$(CUDA_LIBDIR):' >> $@
	@echo '$$LD_LIBRARY_PATH' >> $@
	@echo >> $@
	@echo 'export PATH=$$PATH:$(CUDA_BASE)/bin' >> $@

define TARGET_template
include src/$(1)/Makefile.deps
$(1): $$(foreach dep,$$($(1)_EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) | $(DATA_DEPS)
	+$(MAKE) -C src/$(1)
endef
$(foreach target,$(TARGETS),$(eval $(call TARGET_template,$(target))))

format:
	$(CLANG_FORMAT) -i $(shell find src -name "*.h" -o -name "*.cc" -o -name "*.cu")

clean:
	rm -fR lib obj $(TARGETS)

distclean: | clean
	rm -fR external

dataclean:
	rm -fR data/*.tar.gz data/*.bin data/data_ok

# Data rules
$(DATA_DEPS): $(DATA_TAR_GZ) | $(DATA_BASE)/md5.txt
	cd $(DATA_BASE) && tar zxf $(DATA_TAR_GZ)
	cd $(DATA_BASE) && md5sum *.bin | diff -u md5.txt -
	touch $(DATA_DEPS)

$(DATA_TAR_GZ): | $(DATA_BASE)/url.txt
	curl -o $(DATA_TAR_GZ) $(shell cat $(DATA_BASE)/url.txt)

# External rules
$(EXTERNAL_BASE):
	mkdir -p $@

# TBB
external_tbb: $(TBB_LIB)

$(TBB_BASE):
	git clone --branch 2019_U9 https://github.com/intel/tbb.git $@

$(TBB_LIBDIR): $(TBB_BASE)
	mkdir -p $@

# Let TBB Makefile to define its own CXXFLAGS
$(TBB_LIB): CXXFLAGS:=
$(TBB_LIB): $(TBB_BASE) $(TBB_LIBDIR)
	+$(MAKE) -C $(TBB_BASE) stdver=c++17
	cp $$(find $(TBB_BASE)/build -name *.so*) $(TBB_LIBDIR)

# CUB
external_cub: $(CUB_BASE)

$(CUB_BASE):
	git clone --branch 1.8.0 https://github.com/NVlabs/cub.git $@

# Eigen
external_eigen: $(EIGEN_BASE)

$(EIGEN_BASE):
	git clone https://github.com/cms-externals/eigen-git-mirror $@
	cd $@ && git checkout -b cms_branch d812f411c3f9

# Kokkos
external_kokkos: $(KOKKOS_INSTALL)

$(KOKKOS_SRC):
	git clone --branch 3.0.00 https://github.com/kokkos/kokkos.git $@

$(KOKKOS_BUILD):
	mkdir -p $@

$(KOKKOS_MAKEFILE): $(KOKKOS_BUILD) $(KOKKOS_SRC)
	cd $(KOKKOS_BUILD) && cmake $(KOKKOS_SRC) $(KOKKOS_CMAKEFLAGS)

$(KOKKOS_LIB): $(KOKKOS_MAKEFILE)
	$(MAKE) -C $(KOKKOS_BUILD)
	$(MAKE) -C $(KOKKOS_BUILD) install
