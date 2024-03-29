TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_NAME := $(notdir $(TARGET_DIR))
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(alpakatest_EXTERNAL_DEPENDS)

$(TARGET):
test_cpu: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --maxEvents 2 --serial
	$(TARGET) --maxEvents 2 --tbb
	@echo "Succeeded"
test_nvidiagpu: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --maxEvents 2 --cuda
	@echo "Succeeded"
test_amdgpu: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --maxEvents 2 --hip
	@echo "Succeeded"
test_intelgpu:
test_auto:
.PHONY: test_cpu test_nvidiagpu test_amdgpu test_intelgpu test_auto

EXE_SRC := $(wildcard $(TARGET_DIR)/bin/*.cc)
EXE_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(EXE_SRC:%=%.o))
EXE_DEP := $(EXE_OBJ:$.o=$.d)

LIBNAMES := $(filter-out plugin-% bin test Makefile% plugins.txt%,$(wildcard *))
PLUGINNAMES := $(patsubst plugin-%,%,$(filter plugin-%,$(wildcard *)))
MY_CXXFLAGS := -I$(TARGET_DIR) -DLIB_DIR=$(LIB_DIR)/$(TARGET_NAME) -DALPAKA_HOST_ONLY -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT -DALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT
ifdef CUDA_BASE
MY_CXXFLAGS += -DALPAKA_ACC_GPU_CUDA_PRESENT -DALPAKA_ACC_GPU_CUDA_ONLY_MODE
endif
ifdef ROCM_BASE
MY_CXXFLAGS += -DALPAKA_ACC_GPU_HIP_PRESENT -DALPAKA_ACC_GPU_HIP_ONLY_MODE
endif
MY_LDFLAGS := -Wl,-rpath,$(LIB_DIR)/$(TARGET_NAME) -L$(LIB_DIR)/$(TARGET_NAME)

ALL_DEPENDS := $(EXE_DEP)
# Files for libraries
LIBS :=
define LIB_template
$(1)_SRC := $$(wildcard $(TARGET_DIR)/$(1)/*.cc)
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1).so
LIBS += $$($(1)_LIB)
$(1)_LDFLAGS := -l$(1)
# portable alpaka code
$(1)_PORTABLE_SRC := $$(wildcard $(TARGET_DIR)/$(1)/alpaka/*.cc)
ifneq ($$(strip $$($(1)_PORTABLE_SRC)),)
# serial backend
$(1)_SERIAL_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.serial.o))
$(1)_SERIAL_DEP := $$($(1)_SERIAL_OBJ:$.o=$.d)
$(1)_SERIAL_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1)_serial.so
LIBS += $$($(1)_SERIAL_LIB)
$(1)_SERIAL_LDFLAGS := -l$(1)_serial
# TBB backend
$(1)_TBB_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.tbb.o))
$(1)_TBB_DEP := $$($(1)_TBB_OBJ:$.o=$.d)
$(1)_TBB_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1)_tbb.so
LIBS += $$($(1)_TBB_LIB)
$(1)_TBB_LDFLAGS := -l$(1)_tbb
# CUDA backend
ifdef CUDA_BASE
$(1)_CUDA_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.cuda.o))
$(1)_CUDA_DEP := $$($(1)_CUDA_OBJ:$.o=$.d)
$(1)_CUDA_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1)_cuda.so
LIBS += $$($(1)_CUDA_LIB)
$(1)_CUDA_LDFLAGS := -l$(1)_cuda
#
$(1)_CUOBJ := $$($(1)_CUDA_OBJ)
$(1)_CUDADLINK := $$(if $$(strip $$($(1)_CUOBJ)),$(OBJ_DIR)/$(TARGET_NAME)/$(1)/lib$(1)_cudalink.o)
endif
# ROCm backend
ifdef ROCM_BASE
$(1)_ROCM_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.rocm.o))
$(1)_ROCM_DEP := $$($(1)_ROCM_OBJ:$.o=$.d)
$(1)_ROCM_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1)_rocm.so
LIBS += $$($(1)_ROCM_LIB)
$(1)_ROCM_LDFLAGS := -l$(1)_rocm
endif
endif # if PORTABLE_SRC is not empty
ALL_DEPENDS += $$($(1)_DEP) $$($(1)_SERIAL_DEP) $$($(1)_TBB_DEP) $$($(1)_CUDA_DEP) $$($(1)_ROCM_DEP)
endef
$(foreach lib,$(LIBNAMES),$(eval $(call LIB_template,$(lib))))

# Files for plugins
PLUGINS :=
define PLUGIN_template
$(1)_SRC := $$(wildcard $(TARGET_DIR)/plugin-$(1)/*.cc)
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1).so
PLUGINS += $$($(1)_LIB)
# portable alpaka code
$(1)_PORTABLE_SRC := $$(wildcard $(TARGET_DIR)/plugin-$(1)/alpaka/*.cc)
ifneq ($$(strip $$($(1)_PORTABLE_SRC)),)
# serial backend
$(1)_SERIAL_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.serial.o))
$(1)_SERIAL_DEP := $$($(1)_SERIAL_OBJ:$.o=$.d)
$(1)_SERIAL_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1)_serial.so
PLUGINS += $$($(1)_SERIAL_LIB)
PLUGINNAMES += $(1)_serial
# TBB backend
$(1)_TBB_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.tbb.o))
$(1)_TBB_DEP := $$($(1)_TBB_OBJ:$.o=$.d)
$(1)_TBB_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1)_tbb.so
PLUGINS += $$($(1)_TBB_LIB)
PLUGINNAMES += $(1)_tbb
# CUDA backend
ifdef CUDA_BASE
$(1)_CUDA_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.cuda.o))
$(1)_CUDA_DEP := $$($(1)_CUDA_OBJ:$.o=$.d)
$(1)_CUDA_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1)_cuda.so
PLUGINS += $$($(1)_CUDA_LIB)
PLUGINNAMES += $(1)_cuda
#
$(1)_CUOBJ := $$($(1)_CUDA_OBJ)
$(1)_CUDADLINK := $$(if $$(strip $$($(1)_CUOBJ)),$(OBJ_DIR)/$(TARGET_NAME)/plugin-$(1)/plugin$(1)_cudadlink.o,)
endif
# ROCm backend
ifdef ROCM_BASE
$(1)_ROCM_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.rocm.o))
$(1)_ROCM_DEP := $$($(1)_ROCM_OBJ:$.o=$.d)
$(1)_ROCM_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1)_rocm.so
PLUGINS += $$($(1)_ROCM_LIB)
PLUGINNAMES += $(1)_rocm
endif
endif # if PORTABLE_SRC is not empty
ALL_DEPENDS += $$($(1)_DEP) $$($(1)_SERIAL_DEP) $$($(1)_TBB_DEP) $$($(1)_CUDA_DEP) $$($(1)_ROCM_DEP)
endef
$(foreach lib,$(PLUGINNAMES),$(eval $(call PLUGIN_template,$(lib))))

# Files for unit tests
TESTS_PORTABLE_SRC := $(wildcard $(TARGET_DIR)/test/alpaka/*.cc)
# serial backend
TESTS_SERIAL_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=%.serial.o))
TESTS_SERIAL_DEP := $(TESTS_SERIAL_OBJ:$.o=$.d)
TESTS_SERIAL_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/alpaka/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.serial,$(TESTS_PORTABLE_SRC))
# TBB backend
TESTS_TBB_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=tbb.serial.o))
TESTS_TBB_DEP := $(TESTS_SERIAL_OBJ:$.o=$.d)
TESTS_TBB_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/alpaka/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.tbb,$(TESTS_PORTABLE_SRC))
# CUDA backend
ifdef CUDA_BASE
TESTS_CUDA_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=%.cuda.o))
TESTS_CUDA_DEP := $(TESTS_CUDA_OBJ:$.o=$.d)
TESTS_CUDA_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/alpaka/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.cuda,$(TESTS_PORTABLE_SRC))
TESTS_CUDADLINK := $(TESTS_CUDA_OBJ:$cu.o=$cudadlink.o)
endif
# ROCm backend
ifdef ROCM_BASE
TESTS_ROCM_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=%.rocm.o))
TESTS_ROCM_DEP := $(TESTS_ROCM_OBJ:$.o=$.d)
TESTS_ROCM_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/alpaka/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.rocm,$(TESTS_PORTABLE_SRC))
endif
#
TESTS_EXE := $(TESTS_SERIAL_EXE) $(TESTS_TBB_EXE) $(TESTS_CUDA_EXE) $(TESTS_ROCM_EXE)
ALL_DEPENDS += $(TESTS_SERIAL_DEP) $(TESTS_TBB_DEP) $(TESTS_CUDA_DEP) $(TESTS_ROCM_DEP)
# Needed to keep the unit test object files after building $(TARGET)
.SECONDARY: $(TESTS_SERIAL_OBJ) $(TESTS_TBB_OBJ) $(TESTS_CUDA_OBJ) $(TESTS_CUDADLINK) $(TESTS_ROCM_OBJ)

define RUNTEST_template
run_$(1): $(1)
	@echo
	@echo "Running test $(1)"
	@$(1)
	@echo "Succeeded"
test_$(2): run_$(1)
endef
$(foreach test,$(TESTS_SERIAL_EXE),$(eval $(call RUNTEST_template,$(test),cpu)))
$(foreach test,$(TESTS_TBB_EXE),$(eval $(call RUNTEST_template,$(test),cpu)))
$(foreach test,$(TESTS_CUDA_EXE),$(eval $(call RUNTEST_template,$(test),nvidiagpu)))
$(foreach test,$(TESTS_ROCM_EXE),$(eval $(call RUNTEST_template,$(test),amdgpu)))

-include $(ALL_DEPENDS)

# Build targets
$(LIB_DIR)/$(TARGET_NAME)/plugins.txt: $(PLUGINS)
	nm -A -C -D -P --defined-only $(PLUGINS) | sed -n -e"s#$(LIB_DIR)/$(TARGET_NAME)/\(plugin\w\+\.so\): typeinfo for edm::\(PluginFactory\|ESPluginFactory\)::impl::Maker<\([A-Za-z0-9_:]\+\)> V .* .*#\3 \1#p" | sort > $@

$(TARGET): $(EXE_OBJ) $(LIBS) $(PLUGINS) $(LIB_DIR)/$(TARGET_NAME)/plugins.txt | $(TESTS_EXE)
	# Link all libraries, also the "portable" ones
	$(CXX) $(EXE_OBJ) $(LDFLAGS) $(MY_LDFLAGS) -o $@ $(foreach lib,$(LIBNAMES),$($(lib)_LDFLAGS) $($(lib)_SERIAL_LDFLAGS) $($(lib)_TBB_LDFLAGS) $($(lib)_CUDA_LDFLAGS) $($(lib)_ROCM_LDFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

define BUILD_template
$(OBJ_DIR)/$(2)/%.cc.o: $(SRC_DIR)/$(2)/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD
	@cp $(OBJ_DIR)/$(2)/$$*.cc.d $(OBJ_DIR)/$(2)/$$*.cc.d.tmp; \
	  sed 's#\($(2)/$$*\)\.o[ :]*#\1.o \1.d : #g' < $(OBJ_DIR)/$(2)/$$*.cc.d.tmp > $(OBJ_DIR)/$(2)/$$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$$$//' \
	      -e '/^$$$$/ d' -e 's/$$$$/ :/' -e 's/ *//' < $(OBJ_DIR)/$(2)/$$*.cc.d.tmp >> $(OBJ_DIR)/$(2)/$$*.cc.d; \
	  rm $(OBJ_DIR)/$(2)/$$*.cc.d.tmp

$$($(1)_LIB): $$($(1)_OBJ) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_OBJ) $(LDFLAGS) -shared $(SO_LDFLAGS) $(MY_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) -o $$@

$$($(1)_SERIAL_LIB): $$($(1)_SERIAL_OBJ) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_SERIAL_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_SERIAL_OBJ) $(LDFLAGS) -shared $(SO_LDFLAGS) $(MY_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_SERIAL_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) -o $$@

$$($(1)_TBB_LIB): $$($(1)_TBB_OBJ) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_TBB_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_TBB_OBJ) $(LDFLAGS) -shared $(SO_LDFLAGS) $(MY_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_TBB_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) -o $$@

$$($(1)_CUDA_LIB): $$($(1)_CUDA_OBJ) $$($(1)_CUDADLINK) $$(foreach dep,$(EXTERNAL_DEPENDS_H),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_CUDA_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_CUDA_OBJ) $$($(1)_CUDADLINK) $(LDFLAGS) -shared $(SO_LDFLAGS) $(MY_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_CUDA_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) -o $$@

$$($(1)_ROCM_LIB): $$($(1)_ROCM_OBJ) $$(foreach dep,$(EXTERNAL_DEPENDS_H),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_ROCM_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_ROCM_OBJ) $(LDFLAGS) -shared $(SO_LDFLAGS) $(MY_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_ROCM_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) -o $$@

# Anything depending on Alpaka
# Portable code, for serial backend
$(OBJ_DIR)/$(2)/alpaka/%.cc.serial.o: $(SRC_DIR)/$(2)/alpaka/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD
	@cp $(OBJ_DIR)/$(2)/alpaka/$$*.cc.serial.d $(OBJ_DIR)/$(2)/alpaka/$$*.cc.serial.d.tmp; \
	  sed 's#\($(2)/alpaka/$$*\)\.o[ :]*#\1.o \1.d : #g' < $(OBJ_DIR)/$(2)/alpaka/$$*.cc.serial.d.tmp > $(OBJ_DIR)/$(2)/alpaka/$$*.cc.serial.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$$$//' \
	      -e '/^$$$$/ d' -e 's/$$$$/ :/' -e 's/ *//' < $(OBJ_DIR)/$(2)/alpaka/$$*.cc.serial.d.tmp >> $(OBJ_DIR)/$(2)/alpaka/$$*.cc.serial.d; \
	  rm $(OBJ_DIR)/$(2)/alpaka/$$*.cc.serial.d.tmp

# Portable code, for TBB backend
$(OBJ_DIR)/$(2)/alpaka/%.cc.tbb.o: $(SRC_DIR)/$(2)/alpaka/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD
	@cp $(OBJ_DIR)/$(2)/alpaka/$$*.cc.tbb.d $(OBJ_DIR)/$(2)/alpaka/$$*.cc.tbb.d.tmp; \
	  sed 's#\($(2)/alpaka/$$*\)\.o[ :]*#\1.o \1.d : #g' < $(OBJ_DIR)/$(2)/alpaka/$$*.cc.tbb.d.tmp > $(OBJ_DIR)/$(2)/alpaka/$$*.cc.tbb.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$$$//' \
	      -e '/^$$$$/ d' -e 's/$$$$/ :/' -e 's/ *//' < $(OBJ_DIR)/$(2)/alpaka/$$*.cc.tbb.d.tmp >> $(OBJ_DIR)/$(2)/alpaka/$$*.cc.tbb.d; \
	  rm $(OBJ_DIR)/$(2)/alpaka/$$*.cc.tbb.d.tmp

# Portable code, for CUDA backend
ifdef CUDA_BASE
$(OBJ_DIR)/$(2)/alpaka/%.cc.cuda.o: $(SRC_DIR)/$(2)/alpaka/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CUDA_NVCC) -x cu $(CUDA_CUFLAGS) $(CUDA_CXXFLAGS) $(MY_CXXFLAGS) -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND -UALPAKA_HOST_ONLY $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_NVCC_CXXFLAGS)) -c $$< -o $$@ -MMD

$$($(1)_CUDADLINK): $$($(1)_CUOBJ)
	$(CUDA_NVCC) $(CUDA_DLINKFLAGS) $(CUDA_LDFLAGS) $$($(1)_CUOBJ) -o $$@
endif

# Portable code, for ROCm backend
ifdef ROCM_BASE
$(OBJ_DIR)/$(2)/alpaka/%.cc.rocm.o: $(SRC_DIR)/$(2)/alpaka/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(ROCM_HIPCC) $(HIPCC_CXXFLAGS) $(MY_CXXFLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_ACC_GPU_HIP_ASYNC_BACKEND -UALPAKA_HOST_ONLY $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD
endif

endef

$(foreach lib,$(LIBNAMES),$(eval $(call BUILD_template,$(lib),$(TARGET_NAME)/$(lib))))
$(foreach lib,$(PLUGINNAMES),$(eval $(call BUILD_template,$(lib),$(TARGET_NAME)/plugin-$(lib))))

$(OBJ_DIR)/$(TARGET_NAME)/bin/%.cc.o: $(SRC_DIR)/$(TARGET_NAME)/bin/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD
	@cp $(@D)/$*.cc.d $(@D)/$*.cc.d.tmp; \
	  sed 's#\($(TARGET_NAME)/$*\)\.o[ :]*#\1.o \1.d : #g' < $(@D)/$*.cc.d.tmp > $(@D)/$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' -e 's/ *//' < $(@D)/$*.cc.d.tmp >> $(@D)/$*.cc.d; \
	  rm $(@D)/$*.cc.d.tmp

# Tests, assume all are portable
# Serial backend
$(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.serial.o: $(SRC_DIR)/$(TARGET_NAME)/test/alpaka/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD
	@cp $(@D)/$*.cc.serial.d $(@D)/$*.cc.serial.d.tmp; \
	  sed 's#\($(TARGET_NAME)/$*\)\.o[ :]*#\1.o \1.d : #g' < $(@D)/$*.cc.serial.d.tmp > $(@D)/$*.cc.serial.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' -e 's/ *//' < $(@D)/$*.cc.serial.d.tmp >> $(@D)/$*.cc.serial.d; \
	  rm $(@D)/$*.cc.serial.d.tmp

$(TEST_DIR)/$(TARGET_NAME)/%.serial: $(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.serial.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) $(MY_LDFLAGS) -o $@ $(foreach lib,$(LIBNAMES),$($(lib)_LDFLAGS) $($(lib)_SERIAL_LDFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

# TBB backend
$(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.tbb.o: $(SRC_DIR)/$(TARGET_NAME)/test/alpaka/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD
	@cp $(@D)/$*.cc.tbb.d $(@D)/$*.cc.tbb.d.tmp; \
	  sed 's#\($(TARGET_NAME)/$*\)\.o[ :]*#\1.o \1.d : #g' < $(@D)/$*.cc.tbb.d.tmp > $(@D)/$*.cc.tbb.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' -e 's/ *//' < $(@D)/$*.cc.tbb.d.tmp >> $(@D)/$*.cc.tbb.d; \
	  rm $(@D)/$*.cc.tbb.d.tmp

$(TEST_DIR)/$(TARGET_NAME)/%.tbb: $(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.tbb.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) $(MY_LDFLAGS) -o $@ $(foreach lib,$(LIBNAMES),$($(lib)_LDFLAGS) $($(lib)_TBB_LDFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

# CUDA backend
ifdef CUDA_BASE
$(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.cuda.o: $(SRC_DIR)/$(TARGET_NAME)/test/alpaka/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CUDA_NVCC) -x cu $(CUDA_CUFLAGS) $(CUDA_CXXFLAGS) $(CUDA_TEST_CXXFLAGS) $(MY_CXXFLAGS) -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND -UALPAKA_HOST_ONLY $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_NVCC_CXXFLAGS)) -c $< -o $@ -MMD

$(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cudadlink.o: $(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.o
	$(CUDA_NVCC) $(CUDA_DLINKFLAGS) $(CUDA_LDFLAGS) $< -o $@

$(TEST_DIR)/$(TARGET_NAME)/%.cuda: $(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.cuda.o $(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.cuda.cudadlink.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) $(MY_LDFLAGS) -o $@ $(foreach lib,$(LIBNAMES),$($(lib)_LDFLAGS) $($(lib)_CUDA_LDFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))
endif

# ROCm backend
ifdef ROCM_BASE
$(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.rocm.o: $(SRC_DIR)/$(TARGET_NAME)/test/alpaka/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(ROCM_HIPCC) $(HIPCC_CXXFLAGS) $(ROCM_TEST_CXXFLAGS) $(MY_CXXFLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_ACC_GPU_HIP_ASYNC_BACKEND -UALPAKA_HOST_ONLY $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD

$(TEST_DIR)/$(TARGET_NAME)/%.rocm: $(OBJ_DIR)/$(TARGET_NAME)/test/alpaka/%.cc.rocm.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) $(MY_LDFLAGS) -o $@ $(foreach lib,$(LIBNAMES),$($(lib)_LDFLAGS) $($(lib)_ROCM_LDFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))
endif
