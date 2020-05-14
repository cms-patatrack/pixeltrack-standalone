TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_NAME := $(notdir $(TARGET_DIR))
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(kokkos_EXTERNAL_DEPENDS)

$(TARGET):
test_cpu: $(TARGET)
	$(TARGET) --maxEvents 2 --serial
test_cuda: $(TARGET)
	$(TARGET) --maxEvents 2 --cuda

EXE_SRC := $(wildcard $(TARGET_DIR)/bin/*.cc)
EXE_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(EXE_SRC:%=%.o))
EXE_DEP := $(EXE_OBJ:$.o=$.d)

LIBNAMES := $(filter-out plugin-% bin Makefile% plugins.txt%,$(wildcard *))
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
# CUDA backend
$(1)_CUDA_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_PORTABLE_SRC:%=%.cuda.o))
$(1)_CUDA_DEP := $$($(1)_CUDA_OBJ:$.o=$.d)
# this means all built kokkos objects...
$(1)_CUOBJ := $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ) $$($(1)_CUDA_OBJ)
$(1)_CUDADLINK := $$(if $$(strip $$($(1)_CUOBJ)),$(OBJ_DIR)/$(TARGET_NAME)/plugin-$(1)/plugin$(1)_cudalink.o)

ALL_DEPENDS += $$($(1)_DEP) $$($(1)_HOST_DEP) $$($(1)_SERIAL_DEP) $$($(1)_CUDA_DEP)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1).so
PLUGINS += $$($(1)_LIB)
endef
$(foreach lib,$(PLUGINNAMES),$(eval $(call PLUGIN_template,$(lib))))

-include $(ALL_DEPENDS)

# Build targets
$(TARGET): $(EXE_OBJ) $(LIBS) $(PLUGINS)
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

$$($(1)_LIB): $$($(1)_OBJ) $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ) $$($(1)_CUDA_OBJ) $$($(1)_CUDADLINK) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_OBJ) $$($(1)_HOST_OBJ) $$($(1)_SERIAL_OBJ) $$($(1)_CUDA_OBJ) $$($(1)_CUDADLINK) $(LDFLAGS) -shared $(SO_LDFLAGS) $(LIB_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) $(LIB_LDFLAGS) -o $$@

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

$$($(1)_CUDADLINK): $$($(1)_CUOBJ)
	$(CUDA_NVCC) $(CUDA_DLINKFLAGS) $(CUDA_LDFLAGS) $$($(1)_CUOBJ) -o $$@
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
