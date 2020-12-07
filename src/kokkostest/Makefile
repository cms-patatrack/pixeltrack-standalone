TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_NAME := $(notdir $(TARGET_DIR))
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(kokkostest_EXTERNAL_DEPENDS)

$(TARGET):
test_cpu: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --maxEvents 2 --serial
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
ifneq ($(KOKKOS_DEVICE_PARALLEL),CUDA)
LIBNAMES := $(filter-out CUDACore,$(LIBNAMES))
endif
PLUGINNAMES := $(patsubst plugin-%,%,$(filter plugin-%,$(wildcard *)))
MY_CXXFLAGS := -I$(TARGET_DIR) -DSRC_DIR=$(TARGET_DIR) -DLIB_DIR=$(LIB_DIR)/$(TARGET_NAME)
MY_LDFLAGS := -ldl -Wl,-rpath,$(LIB_DIR)/$(TARGET_NAME)
MY_DEVICE_CXXFLAGS := $(MY_CXXFLAGS)
MY_DEVICE_LDFLAGS := $(MY_LDFLAGS)
ifeq ($(KOKKOS_DEVICE_PARALLEL),CUDA)
MY_DEVICE_CXXFLAGS += -x cu
MY_DEVICE_LDFLAGS := --linker-options '-rpath,$(LIB_DIR)/$(TARGET_NAME)'
endif
LIB_LDFLAGS := -L$(LIB_DIR)/$(TARGET_NAME)

ALL_DEPENDS := $(EXE_DEP)
# Files for libraries
LIBS :=
define LIB_template
# host code
$(1)_SRC := $$(wildcard $(TARGET_DIR)/$(1)/*.cc)
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
# kokkos code, built only for host
$(1)_HOST_SRC := $$(wildcard $(TARGET_DIR)/$(1)/kokkoshost/*.cc)
$(1)_HOST_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_HOST_SRC:%=%.o))
$(1)_HOST_DEP := $$($(1)_HOST_OBJ:$.o=$.d)
# portable kokkos code
$(1)_PORTABLE_SRC := $$(wildcard $(TARGET_DIR)/$(1)/kokkos/*.cc)
# serial backend
$(1)_SERIAL_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.serial.o))
$(1)_SERIAL_DEP := $$($(1)_SERIAL_OBJ:$.o=$.d)
# host parallel backend
$(1)_PARALLEL_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.parallel.o))
$(1)_PARALLEL_DEP := $$($(1)_PARALLEL_OBJ:$.o=$.d)
# device parallel backend
$(1)_DEVICE_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.device.o))
$(1)_DEVICE_DEP := $$($(1)_DEVICE_OBJ:$.o=$.d)

# this means all built kokkos objects...
$(1)_DEVOBJ := $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ)
ALL_DEPENDS += $$($(1)_DEP) $$($(1)_HOST_DEP) $$($(1)_SERIAL_DEP)
ifdef KOKKOS_HOST_PARALLEL
$(1)_DEVOBJ += $$($(1)_PARALLEL_OBJ)
ALL_DEPENDS += $$((1)_PARALLEL_DEP)
endif
ifdef KOKKOS_DEVICE_PARALLEL
$(1)_DEVOBJ += $$($(1)_DEVICE_OBJ)
ALL_DEPENDS += $$($(1)_DEVICE_DEP)
endif

$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1).so
LIBS += $$($(1)_LIB)
$(1)_LDFLAGS := -l$(1)
endef
$(foreach lib,$(LIBNAMES),$(eval $(call LIB_template,$(lib))))

# Files for plugins
PLUGINS :=
define PLUGIN_template
# host code
$(1)_SRC := $$(wildcard $(TARGET_DIR)/plugin-$(1)/*.cc)
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
# kokkos code, built only for host
$(1)_HOST_SRC := $$(wildcard $(TARGET_DIR)/plugin-$(1)/kokkoshost/*.cc)
$(1)_HOST_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_HOST_SRC:%=%.o))
$(1)_HOST_DEP := $$($(1)_HOST_OBJ:$.o=$.d)
# portable kokkos code
$(1)_PORTABLE_SRC := $$(wildcard $(TARGET_DIR)/plugin-$(1)/kokkos/*.cc)
# serial backend
$(1)_SERIAL_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.serial.o))
$(1)_SERIAL_DEP := $$($(1)_SERIAL_OBJ:$.o=$.d)
# host parallel backend
$(1)_PARALLEL_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.parallel.o))
$(1)_PARALLEL_DEP := $$($(1)_PARALLEL_OBJ:$.o=$.d)
# device parallel backend
$(1)_DEVICE_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.device.o))
$(1)_DEVICE_DEP := $$($(1)_DEVICE_OBJ:$.o=$.d)

# this means all built kokkos objects...
$(1)_DEVOBJ := $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ)
ALL_DEPENDS += $$($(1)_DEP) $$($(1)_HOST_DEP) $$($(1)_SERIAL_DEP)
ifdef KOKKOS_HOST_PARALLEL
$(1)_DEVOBJ += $$($(1)_PARALLEL_OBJ)
ALL_DEPENDS += $$((1)_PARALLEL_DEP)
endif
ifdef KOKKOS_DEVICE_PARALLEL
$(1)_DEVOBJ += $$($(1)_DEVICE_OBJ)
ALL_DEPENDS += $$($(1)_DEVICE_DEP)
endif

$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1).so
PLUGINS += $$($(1)_LIB)
endef
$(foreach lib,$(PLUGINNAMES),$(eval $(call PLUGIN_template,$(lib))))

# Files for unit tests
TESTS_PORTABLE_SRC := $(wildcard $(TARGET_DIR)/test/kokkos/*.cc)
# serial backend
TESTS_SERIAL_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=%.serial.o))
TESTS_SERIAL_DEP := $(TESTS_SERIAL_OBJ:$.o=$.d)
TESTS_SERIAL_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.serial,$(TESTS_PORTABLE_SRC))
# host parallel backend
TESTS_PARALLEL_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=%.parallel.o))
TESTS_PARALLEL_DEP := $(TESTS_PARALLEL_OBJ:$.o=$.d)
TESTS_PARALLEL_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.parallel,$(TESTS_PORTABLE_SRC))
# device parallel backend
TESTS_DEVICE_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=%.device.o))
TESTS_DEVICE_DEP := $(TESTS_DEVICE_OBJ:$.o=$.d)
TESTS_DEVICE_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.device,$(TESTS_PORTABLE_SRC))
#
TESTS_EXE := $(TESTS_SERIAL_EXE)
TESTS_DEVOBJ := $(TESTS_SERIAL_OBJ)
ALL_DEPENDS += $(TESTS_SERIAL_DEP)
ifdef KOKKOS_HOST_PARALLEL
TESTS_EXE += $(TESTS_PARALLEL_EXE)
TESTS_DEVOBJ += $(TESTS_PARALLEL_OBJ)
ALL_DEPENDS += $(TESTS_PARALLEL_DEP)
endif
ifdef KOKKOS_DEVICE_PARALLEL
TESTS_EXE += $(TESTS_DEVICE_EXE)
TESTS_DEVOBJ += $(TESTS_DEVICE_OBJ)
ALL_DEPENDS += $(TESTS_PARALLEL_DEP)
endif

# Needed to keep the unit test object files after building $(TARGET)
.SECONDARY: $(TESTS_DEVOBJ)

define RUNTEST_template
run_$(1): $(1)
	@echo
	@echo "Running test $(1)"
	@$(1)
	@echo "Succeeded"
test_$(2): run_$(1)
endef
$(foreach test,$(TESTS_SERIAL_EXE),$(eval $(call RUNTEST_template,$(test),cpu)))
ifdef KOKKOS_HOST_PARALLEL
$(foreach test,$(TESTS_PARALLEL_EXE),$(eval $(call RUNTEST_template,$(test),cpu)))
endif
ifeq ($(KOKKOS_DEVICE_PARALLEL),CUDA)
$(foreach test,$(TESTS_DEVICE_EXE),$(eval $(call RUNTEST_template,$(test),nvidiagpu)))
endif
ifeq ($(KOKKOS_DEVICE_PARALLEL),HIP)
$(foreach test,$(TESTS_DEVICE_EXE),$(eval $(call RUNTEST_template,$(test),amdgpu)))
endif

-include $(ALL_DEPENDS)

# Build targets
$(TARGET): $(EXE_OBJ) $(LIBS) $(PLUGINS) | $(TESTS_EXE)
	$(CXX) $(EXE_OBJ) $(LDFLAGS) $(MY_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

define BUILD_template
$(OBJ_DIR)/$(2)/%.cc.o: $(SRC_DIR)/$(2)/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD
	@cp $(OBJ_DIR)/$(2)/$$*.cc.d $(OBJ_DIR)/$(2)/$$*.cc.d.tmp; \
	  sed 's#\($(2)/$$*\)\.o[ :]*#\1.o \1.d : #g' < $(OBJ_DIR)/$(2)/$$*.cc.d.tmp > $(OBJ_DIR)/$(2)/$$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$$$//' \
	      -e '/^$$$$/ d' -e 's/$$$$/ :/' -e 's/ *//' < $(OBJ_DIR)/$(2)/$$*.cc.d.tmp >> $(OBJ_DIR)/$(2)/$$*.cc.d; \
	  rm $(OBJ_DIR)/$(2)/$$*.cc.d.tmp

ifeq ($$(strip $$($(1)_DEVOBJ)),)
$$($(1)_LIB): $$($(1)_OBJ) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_OBJ) $(LDFLAGS) -shared $(SO_LDFLAGS) $(LIB_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) $(LIB_LDFLAGS) -o $$@
else
$$($(1)_LIB): $$($(1)_OBJ) $$($(1)_DEVOBJ) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(KOKKOS_DEVICE_CXX) $$($(1)_OBJ) $$($(1)_DEVOBJ) $(KOKKOS_DEVICE_LDFLAGS) -shared $(KOKKOS_DEVICE_SO_LDFLAGS) $(LIB_LDFLAGS) $(MY_DEVICE_LDFLAGS)  $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) $(LIB_LDFLAGS) -o $$@
endif

# Anything depending on Kokkos
# Host-only code, need a special rule because have to build with nvcc if CUDA backend is enabled
$(OBJ_DIR)/$(2)/kokkoshost/%.cc.o: $(SRC_DIR)/$(2)/kokkoshost/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(KOKKOS_DEVICE_CXX) $(KOKKOS_DEVICE_CXXFLAGS) $(MY_DEVICE_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD

# Portable code, for serial backend
$(OBJ_DIR)/$(2)/kokkos/%.cc.serial.o: $(SRC_DIR)/$(2)/kokkos/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(KOKKOS_DEVICE_CXX) -DKOKKOS_BACKEND_SERIAL $(KOKKOS_DEVICE_CXXFLAGS) $(MY_DEVICE_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD

# Portable code, for host parallel backend
ifdef KOKKOS_HOST_PARALLEL
$(OBJ_DIR)/$(2)/kokkos/%.cc.parallel.o: $(SRC_DIR)/$(2)/kokkos/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(KOKKOS_DEVICE_CXX) -DKOKKOS_BACKEND_$(KOKKOS_HOST_PARALLEL) $(KOKKOS_DEVICE_CXXFLAGS) $(MY_DEVICE_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD
endif

# Portable code, for device parallel backend
ifdef KOKKOS_DEVICE_PARALLEL
$(OBJ_DIR)/$(2)/kokkos/%.cc.device.o: $(SRC_DIR)/$(2)/kokkos/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(KOKKOS_DEVICE_CXX) -DKOKKOS_BACKEND_$(KOKKOS_DEVICE_PARALLEL) $(KOKKOS_DEVICE_CXXFLAGS) $(MY_DEVICE_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD
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
$(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.serial.o: $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(KOKKOS_DEVICE_CXX) -DKOKKOS_BACKEND_SERIAL $(KOKKOS_DEVICE_CXXFLAGS) $(MY_DEVICE_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD

$(TEST_DIR)/$(TARGET_NAME)/%.serial: $(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.serial.o| $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(KOKKOS_DEVICE_CXX) $^ $(KOKKOS_DEVICE_LDFLAGS) $(MY_DEVICE_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

# Host parallel backend
ifdef KOKKOS_HOST_PARALLEL
$(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.parallel.o: $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(KOKKOS_DEVICE_CXX) -DKOKKOS_BACKEND_$(KOKKOS_HOST_PARALLEL) $(KOKKOS_DEVICE_CXXFLAGS) $(MY_DEVICE_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD

$(TEST_DIR)/$(TARGET_NAME)/%.parallel: $(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.parallel.o| $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(KOKKOS_DEVICE_CXX) $^ $(KOKKOS_DEVICE_LDFLAGS) $(MY_DEVICE_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))
endif

# Device parallel backend
ifdef KOKKOS_DEVICE_PARALLEL
$(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.device.o: $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(KOKKOS_DEVICE_CXX) -DKOKKOS_BACKEND_$(KOKKOS_DEVICE_PARALLEL) $(KOKKOS_DEVICE_CXXFLAGS) $(MY_DEVICE_CXXFLAGS) $(KOKKOS_DEVICE_TEST_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD

$(TEST_DIR)/$(TARGET_NAME)/%.device: $(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.device.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(KOKKOS_DEVICE_CXX) $^ $(KOKKOS_DEVICE_LDFLAGS) $(MY_DEVICE_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))
endif
