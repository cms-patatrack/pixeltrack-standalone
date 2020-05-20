export BASE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Build flags
export CXX := g++
USER_CXXFLAGS :=
HOST_CXXFLAGS := -O2 -fPIC -fdiagnostics-show-option -felide-constructors -fmessage-length=0 -fno-math-errno -ftree-vectorize -fvisibility-inlines-hidden --param vect-max-version-for-alias-checks=50 -msse3 -pipe -pthread -Xassembler --compress-debug-sections -Werror=address -Wall -Werror=array-bounds -Wno-attributes -Werror=conversion-null -Werror=delete-non-virtual-dtor -Wno-deprecated -Werror=format-contains-nul -Werror=format -Wno-long-long -Werror=main -Werror=missing-braces -Werror=narrowing -Wno-non-template-friend -Wnon-virtual-dtor -Werror=overflow -Werror=overlength-strings -Wparentheses -Werror=pointer-arith -Wno-psabi -Werror=reorder -Werror=return-local-addr -Wreturn-type -Werror=return-type -Werror=sign-compare -Werror=strict-aliasing -Wstrict-overflow -Werror=switch -Werror=type-limits -Wunused -Werror=unused-but-set-variable -Wno-unused-local-typedefs -Werror=unused-value -Wno-error=unused-variable -Wno-vla -Werror=write-strings
export CXXFLAGS := -std=c++17 $(HOST_CXXFLAGS) $(USER_CXXFLAGS)
export LDFLAGS := -pthread -Wl,-E -lstdc++fs
export LDFLAGS_NVCC := --linker-options '-E' --linker-options '-lstdc++fs'
export SO_LDFLAGS := -Wl,-z,defs
export SO_LDFLAGS_NVCC := --linker-options '-z,defs'

CLANG_FORMAT := clang-format-8
CMAKE := cmake

# Source code
export SRC_DIR := $(BASE_DIR)/src

# Directory where to put object and dependency files
export OBJ_DIR := $(BASE_DIR)/obj

# Directory where to put libraries
export LIB_DIR := $(BASE_DIR)/lib

# Directory where to put unit test executables
export TEST_DIR := $(BASE_DIR)/test

# System external definitions
CUDA_BASE := /usr/local/cuda
CUDA_LIBDIR := $(CUDA_BASE)/lib64
USER_CUDAFLAGS :=
export CUDA_DEPS := $(CUDA_BASE)/lib64/libcudart.so
export CUDA_ARCH := 35 60 70
export CUDA_CXXFLAGS := -I$(CUDA_BASE)/include
export CUDA_LDFLAGS := -L$(CUDA_BASE)/lib64 -lcudart -lcudadevrt
export CUDA_NVCC := $(CUDA_BASE)/bin/nvcc
define CUFLAGS_template
$(2)NVCC_FLAGS := $$(foreach ARCH,$(1),-gencode arch=compute_$$(ARCH),code=sm_$$(ARCH)) --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info --source-in-ptx --cudart=shared
$(2)NVCC_COMMON := -std=c++14 -O3 $$($(2)NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS) $(USER_CXXFLAGS)'
$(2)CUDA_CUFLAGS := -dc $$($(2)NVCC_COMMON) $(USER_CUDAFLAGS)
$(2)CUDA_DLINKFLAGS := -dlink $$($(2)NVCC_COMMON)
endef
$(eval $(call CUFLAGS_template,$(CUDA_ARCH),))
export CUDA_CUFLAGS
export CUDA_DLINKFLAGS

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
export EIGEN_CXXFLAGS := -I$(EIGEN_BASE) -DEIGEN_DONT_PARALLELIZE
export EIGEN_LDFLAGS :=

BOOST_BASE := /usr
# Minimum required version of Boost, e.g. 1.65.1
BOOST_MIN_VERSION := 106501
# Check if an external version of Boost is present and recent enough
ifeq ($(wildcard $(BOOST_BASE)/include/boost/version.hpp),)
NEED_BOOST := true
else
NEED_BOOST := $(shell awk '/\#define BOOST_VERSION\>/ { if ($$3 < $(BOOST_MIN_VERSION)) print "true" }' $(BOOST_BASE)/include/boost/version.hpp )
endif
ifeq ($(NEED_BOOST),true)
BOOST_BASE := $(EXTERNAL_BASE)/boost
endif
export BOOST_DEPS := $(BOOST_BASE)
export BOOST_CXXFLAGS := -I$(BOOST_BASE)/include
export BOOST_LDFLAGS := -L$(BOOST_BASE)/lib

ALPAKA_BASE := $(EXTERNAL_BASE)/alpaka
export ALPAKA_DEPS := $(ALPAKA_BASE)
export ALPAKA_CXXFLAGS := -I$(ALPAKA_BASE)/include
export ALPAKA_CUFLAGS := $(CUDA_CUFLAGS) -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored

CUPLA_BASE := $(EXTERNAL_BASE)/cupla
export CUPLA_DEPS := $(CUPLA_BASE)/lib
export CUPLA_LIBDIR := $(CUPLA_BASE)/lib
export CUPLA_CXXFLAGS := -I$(CUPLA_BASE)/include
export CUPLA_LDFLAGS := -L$(CUPLA_LIBDIR)

KOKKOS_BASE := $(EXTERNAL_BASE)/kokkos
KOKKOS_SRC := $(KOKKOS_BASE)/source
KOKKOS_BUILD := $(KOKKOS_BASE)/build
export KOKKOS_INSTALL := $(KOKKOS_BASE)/install
KOKKOS_LIBDIR := $(KOKKOS_INSTALL)/lib
export KOKKOS_LIB := $(KOKKOS_LIBDIR)/libkokkoscore.a
KOKKOS_MAKEFILE := $(KOKKOS_BUILD)/Makefile
KOKKOS_CMAKEFLAGS := -DCMAKE_INSTALL_PREFIX=$(KOKKOS_INSTALL) \
                     -DCMAKE_INSTALL_LIBDIR=lib \
                     -DKokkos_CXX_STANDARD=14 \
                     -DCMAKE_CXX_COMPILER=$(KOKKOS_SRC)/bin/nvcc_wrapper -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_CUDA_DIR=$(CUDA_BASE) -DKokkos_ARCH_VOLTA70=On
# if without CUDA, replace the above line with
#                     -DCMAKE_CXX_COMPILER=g++
export KOKKOS_DEPS := $(KOKKOS_LIB)
export KOKKOS_CXXFLAGS := -I$(KOKKOS_INSTALL)/include
KOKKOS_CUDA_ARCH := 70
$(eval $(call CUFLAGS_template,$(KOKKOS_CUDA_ARCH),KOKKOS_))
KOKKOS_CUDA_CUFLAGS := $(KOKKOS_NVCC_COMMON) $(USER_CUDAFLAGS)
export KOKKOS_CUFLAGS := $(KOKKOS_CUDA_CUFLAGS) -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
export KOKKOS_LDFLAGS := -L$(KOKKOS_INSTALL)/lib -lkokkoscore -ldl
export KOKKOS_DLINKFLAGS := $(KOKKOS_CUDA_DLINKFLAGS)
export NVCC_WRAPPER_DEFAULT_COMPILER := $(CXX)

# force the recreation of the environment file any time the Makefile is updated, before building any other target
-include environment

# Targets and their dependencies on externals
TARGETS := $(notdir $(wildcard $(SRC_DIR)/*))
TEST_CPU_TARGETS := $(patsubst %,test_%_cpu,$(TARGETS))
TEST_CUDA_TARGETS := $(patsubst %,test_%_cuda,$(TARGETS))
all: $(TARGETS)
test: test_cpu test_cuda
test_cpu: $(TEST_CPU_TARGETS)
test_cuda: $(TEST_CUDA_TARGETS)
# $(TARGETS) needs to be PHONY because only the called Makefile knows their dependencies
.PHONY: all $(TARGETS) test test_cpu test_cuda $(TEST_CPU_TARGETS) $(TEST_CUDA_TARGETS)
.PHONY: environment format clean distclean dataclean external_tbb external_cub external_eigen external_kokkos external_kokkos_clean

environment: env.sh
env.sh: Makefile
	@echo '#! /bin/bash'                                                    > $@
	@echo 'if [ -f .original_env ]; then'                                   >> $@
	@echo '  source .original_env'                                          >> $@
	@echo 'else'                                                            >> $@
	@echo '  echo "#! /bin/bash"                       >  .original_env'    >> $@
	@echo '  echo "PATH=$$PATH"                         >> .original_env'   >> $@
	@echo '  echo "LD_LIBRARY_PATH=$$LD_LIBRARY_PATH"   >> .original_env'   >> $@
	@echo 'fi'                                                              >> $@
	@echo                                                                   >> $@
	@echo -n 'export LD_LIBRARY_PATH='                                      >> $@
	@echo -n '$(TBB_LIBDIR):'                                               >> $@
	@echo -n '$(CUDA_LIBDIR):'                                              >> $@
	@echo -n '$(CUPLA_LIBDIR):'                                             >> $@
	@echo -n '$(KOKKOS_LIBDIR):'                                            >> $@
	@echo '$$LD_LIBRARY_PATH'                                               >> $@
	@echo 'export PATH=$$PATH:$(CUDA_BASE)/bin'                             >> $@

define TARGET_template
include src/$(1)/Makefile.deps
$(1): $$(foreach dep,$$($(1)_EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) | $(DATA_DEPS)
	+$(MAKE) -C src/$(1)

test_$(1)_cpu: $(1)
	@echo
	@echo "Testing $(1) for CPU backend"
	+$(MAKE) -C src/$(1) test_cpu
	@echo

test_$(1)_cuda: $(1)
	@echo
	@echo "Testing $(1) for CUDA backend"
	+$(MAKE) -C src/$(1) test_cuda
	@echo
endef
$(foreach target,$(TARGETS),$(eval $(call TARGET_template,$(target))))

format:
	$(CLANG_FORMAT) -i $(shell find src -name "*.h" -o -name "*.cc" -o -name "*.cu")

clean:
	rm -fR lib obj test $(TARGETS)

distclean: | clean
	rm -fR external .original_env

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

# Boost
.PHONY: external_boost
external_boost: $(BOOST_BASE)

# Let Boost define its own CXXFLAGS
$(BOOST_BASE): CXXFLAGS:=
$(BOOST_BASE):
	$(eval BOOST_TMP := $(shell mktemp -d))
	wget -nv https://dl.bintray.com/boostorg/release/1.73.0/source/boost_1_73_0.tar.bz2 -O - | tar xj -C $(BOOST_TMP)
	cd $(BOOST_TMP)/boost_1_73_0 && ./bootstrap.sh && ./b2 install --prefix=$@
	@rm -rf $(BOOST_TMP)
	$(eval undefine BOOST_TMP)

# Alpaka
.PHONY: external_alpaka
external_alpaka: $(ALPAKA_BASE)

$(ALPAKA_BASE):
	git clone git@github.com:alpaka-group/alpaka.git -b release-0.4.1 $@

# Cupla
.PHONY: external_cupla
external_cupla: $(CUPLA_BASE)/lib

$(CUPLA_BASE):
	git clone git@github.com:alpaka-group/cupla.git -b master $@
	cd $@ && git reset --hard 0.2.0
	cd $@ && git config core.sparsecheckout true && /usr/bin/echo -e '/*\n!/alpaka\n!/build\n!/lib' > .git/info/sparse-checkout && git read-tree -v -mu HEAD

$(CUPLA_BASE)/lib: $(CUPLA_BASE) $(ALPAKA_DEPS) $(BOOST_DEPS) $(TBB_DEPS) $(CUDA_DEPS)
	$(MAKE) -C $(CUPLA_BASE) -f $(BASE_DIR)/Makefile.cupla CXX=$(CXX) CUDA_BASE=$(CUDA_BASE) BOOST_BASE=$(BOOST_BASE) TBB_BASE=$(TBB_BASE) ALPAKA_BASE=$(ALPAKA_BASE)

# Kokkos
external_kokkos: $(KOKKOS_LIB)

$(KOKKOS_SRC):
	git clone --branch 3.0.00 https://github.com/kokkos/kokkos.git $@

$(KOKKOS_BUILD):
	mkdir -p $@

$(KOKKOS_MAKEFILE): $(KOKKOS_SRC) | $(KOKKOS_BUILD)
	cd $(KOKKOS_BUILD) && $(CMAKE) $(KOKKOS_SRC) $(KOKKOS_CMAKEFLAGS)

$(KOKKOS_LIB): $(KOKKOS_MAKEFILE)
	$(MAKE) -C $(KOKKOS_BUILD)
	$(MAKE) -C $(KOKKOS_BUILD) install

external_kokkos_clean:
	rm -fR $(KOKKOS_BUILD) $(KOKKOS_INSTALL)
