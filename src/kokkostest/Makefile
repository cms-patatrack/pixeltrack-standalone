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
test_cuda: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --maxEvents 2 --cuda
	@echo "Succeeded"
.PHONY: test_cpu test_cuda

EXE_SRC := $(wildcard $(TARGET_DIR)/bin/*.cc)
EXE_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(EXE_SRC:%=%.o))
EXE_DEP := $(EXE_OBJ:$.o=$.d)

LIBNAMES := $(filter-out plugin-% bin test Makefile% plugins.txt%,$(wildcard *))
PLUGINNAMES := $(patsubst plugin-%,%,$(filter plugin-%,$(wildcard *)))
MY_CXXFLAGS := -I$(TARGET_DIR) -DSRC_DIR=$(TARGET_DIR) -DLIB_DIR=$(LIB_DIR)/$(TARGET_NAME)
MY_LDFLAGS := -ldl -Wl,-rpath,$(LIB_DIR)/$(TARGET_NAME)
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
# CUDA backend
$(1)_CUDA_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.cuda.o))
$(1)_CUDA_DEP := $$($(1)_CUDA_OBJ:$.o=$.d)
# this means all built kokkos objects...
$(1)_CUOBJ := $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ) $$($(1)_CUDA_OBJ)
$(1)_CUDADLINK := $$(if $$(strip $$($(1)_CUOBJ)),$(OBJ_DIR)/$(TARGET_NAME)/$(1)/lib$(1)_cudalink.o)

ALL_DEPENDS += $$($(1)_DEP) $$($(1)_HOST_DEP) $$($(1)_SERIAL_DEP) $$($(1)_CUDA_DEP)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1).so
LIBS += $$($(1)_LIB)
$(1)_LDFLAGS := -l$(1)
endef
$(foreach lib,$(LIBNAMES),$(eval $(call LIB_template,$(lib))))

# Files for plugins
PLUGINS := $(LIB_DIR)/$(TARGET_NAME)/pluginKokkos.so
PLUGIN_OBJ :=
PLUGIN_CUOBJ :=
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
# CUDA backend
$(1)_CUDA_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.cuda.o))
$(1)_CUDA_DEP := $$($(1)_CUDA_OBJ:$.o=$.d)
# this means all built kokkos objects...
$(1)_CUOBJ := $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ) $$($(1)_CUDA_OBJ)
#$(1)_CUDADLINK := $$(if $$(strip $$($(1)_CUOBJ)),$(OBJ_DIR)/$(TARGET_NAME)/plugin-$(1)/plugin$(1)_cudalink.o)

PLUGIN_OBJ += $$($(1)_OBJ)
PLUGIN_CUOBJ += $$($(1)_CUOBJ)

ALL_DEPENDS += $$($(1)_DEP) $$($(1)_HOST_DEP) $$($(1)_SERIAL_DEP) $$($(1)_CUDA_DEP)
#$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1).so
#PLUGINS += $$($(1)_LIB)
endef
$(foreach lib,$(PLUGINNAMES),$(eval $(call PLUGIN_template,$(lib))))
PLUGIN_CUDADLINK := $(if $(strip $(PLUGIN_CUOBJ)),$(OBJ_DIR)/$(TARGET_NAME)/pluginKokkos_cudalink.o)

# Files for unit tests
TESTS_PORTABLE_SRC := $(wildcard $(TARGET_DIR)/test/kokkos/*.cc)
# serial backend
TESTS_SERIAL_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=%.serial.o))
TESTS_SERIAL_DEP := $(TESTS_SERIAL_OBJ:$.o=$.d)
TESTS_SERIAL_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.serial,$(TESTS_PORTABLE_SRC))
# CUDA backend
TESTS_CUDA_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_PORTABLE_SRC:%=%.cuda.o))
TESTS_CUDA_DEP := $(TESTS_CUDA_OBJ:$.o=$.d)
TESTS_CUDA_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%.cuda,$(TESTS_PORTABLE_SRC))
#
TESTS_EXE := $(TESTS_SERIAL_EXE) $(TESTS_CUDA_EXE)
TESTS_CUOBJ := $(TESTS_SERIAL_OBJ) $(TESTS_CUDA_OBJ)
TESTS_CUDADLINK := $(TESTS_CUOBJ:$cu.o=$cudadlink.o)
ALL_DEPENDS += $(TESTS_SERIAL_DEP) $(TESTS_CUDA_DEP)
# Needed to keep the unit test object files after building $(TARGET)
.SECONDARY: $(TESTS_CUOBJ) $(TESTS_CUDADLINK)

define RUNTEST_template
run_$(1): $(1)
	@echo
	@echo "Running test $(1)"
	@$(1)
	@echo "Succeeded"
test_$(2): run_$(1)
endef
$(foreach test,$(TESTS_SERIAL_EXE),$(eval $(call RUNTEST_template,$(test),cpu)))
$(foreach test,$(TESTS_CUDA_EXE),$(eval $(call RUNTEST_template,$(test),cuda)))

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

# Anything depending on Kokkos
# Host-only code, need a special rule because have to build with nvcc if CUDA backend is enabled
$(OBJ_DIR)/$(2)/kokkoshost/%.cc.o: $(SRC_DIR)/$(2)/kokkoshost/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CUDA_NVCC) -x cu $(KOKKOS_CUFLAGS) $(CUDA_CXXFLAGS) $(MY_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD

# Portable code, for serial backend
$(OBJ_DIR)/$(2)/kokkos/%.cc.serial.o: $(SRC_DIR)/$(2)/kokkos/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CUDA_NVCC) -x cu -DKOKKOS_BACKEND_SERIAL $(KOKKOS_CUFLAGS) $(CUDA_CXXFLAGS) $(MY_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD

# Portable code, for CUDA backend
$(OBJ_DIR)/$(2)/kokkos/%.cc.cuda.o: $(SRC_DIR)/$(2)/kokkos/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CUDA_NVCC) -x cu -DKOKKOS_BACKEND_CUDA $(KOKKOS_CUFLAGS) $(CUDA_CXXFLAGS) $(MY_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) -c $$< -o $$@ -MMD
endef

define LIB_template
$$($(1)_LIB): $$($(1)_OBJ) $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ) $$($(1)_CUDA_OBJ) $$($(1)_CUDADLINK) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_OBJ) $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ) $$($(1)_CUDA_OBJ) $$($(1)_CUDADLINK) $(LDFLAGS) -shared $(SO_LDFLAGS) $(LIB_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) $(LIB_LDFLAGS) -o $$@
endef

$(foreach lib,$(LIBNAMES),$(eval $(call BUILD_template,$(lib),$(TARGET_NAME)/$(lib))))
$(foreach lib,$(LIBNAMES),$(eval $(call LIB_template,$(lib),$(TARGET_NAME)/$(lib))))
$(foreach lib,$(PLUGINNAMES),$(eval $(call BUILD_template,$(lib),$(TARGET_NAME)/plugin-$(lib))))

$(PLUGINS): $(PLUGIN_OBJ) $(PLUGIN_CUOBJ) $(PLUGIN_CUDADLINK) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_DEPS)) $(foreach lib,$(PLUGIN_DEPENDS),$($(lib)_LIB))
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $(PLUGIN_OBJ) $(PLUGIN_CUOBJ) $(PLUGIN_CUDADLINK) $(LDFLAGS) -shared $(SO_LDFLAGS) $(LIB_LDFLAGS) $(foreach lib,$(PLUGIN_DEPENDS),$($(lib)_LDFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS)) -o $@

$(PLUGIN_CUDADLINK): $(PLUGIN_CUOBJ)
	$(CUDA_NVCC) $(KOKKOS_DLINKFLAGS) $(PLUGIN_CUOBJ) $(KOKKOS_LDFLAGS) $(CUDA_LDFLAGS) -o $@


$(OBJ_DIR)/$(TARGET_NAME)/bin/%.cc.o: $(SRC_DIR)/$(TARGET_NAME)/bin/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD
	@cp $(@D)/$*.cc.d $(@D)/$*.cc.d.tmp; \
	  sed 's#\($(TARGET_NAME)/$*\)\.o[ :]*#\1.o \1.d : #g' < $(@D)/$*.cc.d.tmp > $(@D)/$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' -e 's/ *//' < $(@D)/$*.cc.d.tmp >> $(@D)/$*.cc.d; \
	  rm $(@D)/$*.cc.d.tmp

# Tests, assume all are portable
# device link rule can be shared
$(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cudadlink.o: $(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.o
	$(CUDA_NVCC) $(KOKKOS_DLINKFLAGS) $< $(KOKKOS_LDFLAGS) $(CUDA_LDFLAGS) -o $@

# Serial backend
$(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.serial.o: $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CUDA_NVCC) -x cu -DKOKKOS_BACKEND_SERIAL $(KOKKOS_CUFLAGS) $(CUDA_CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD

$(TEST_DIR)/$(TARGET_NAME)/%.serial: $(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.serial.o $(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.serial.cudadlink.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) $(MY_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

# CUDA backend
$(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.cuda.o: $(SRC_DIR)/$(TARGET_NAME)/test/kokkos/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CUDA_NVCC) -x cu -DKOKKOS_BACKEND_CUDA $(KOKKOS_CUFLAGS) $(CUDA_CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD

$(TEST_DIR)/$(TARGET_NAME)/%.cuda: $(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.cuda.o $(OBJ_DIR)/$(TARGET_NAME)/test/kokkos/%.cc.cuda.cudadlink.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) $(MY_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))
