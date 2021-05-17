TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_NAME := $(notdir $(TARGET_DIR))
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(cudacompat_EXTERNAL_DEPENDS)

$(TARGET):
test_cpu:
test_nvidiagpu: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --maxEvents 2
	@echo "Succeeded"
test_intelagpu:
test_auto:
.PHONY: test_cpu test_nvidiagpu test_intelgpu test_auto

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
$(1)_SRC := $$(wildcard $(TARGET_DIR)/$(1)/*.cc)
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
ALL_DEPENDS += $$($(1)_DEP)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1).so
LIBS += $$($(1)_LIB)
$(1)_LDFLAGS := -l$(1)
endef
$(foreach lib,$(LIBNAMES),$(eval $(call LIB_template,$(lib))))

# Files for plugins
PLUGINS :=
define PLUGIN_template
$(1)_SRC := $$(wildcard $(TARGET_DIR)/plugin-$(1)/*.cc)
$(1)_CUSRC := $$(wildcard $(TARGET_DIR)/plugin-$(1)/*.cu)
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_CUOBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_CUSRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
ALL_DEPENDS += $$($(1)_DEP)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1).so
PLUGINS += $$($(1)_LIB)
$(1)_CUDADLINK := $$(if $$(strip $$($(1)_CUOBJ)),$(OBJ_DIR)/$(TARGET_NAME)/plugin-$(1)/plugin$(1)_cudadlink.o,)
endef
$(foreach lib,$(PLUGINNAMES),$(eval $(call PLUGIN_template,$(lib))))

# Files for unit tests
TESTS_SRC := $(wildcard $(TARGET_DIR)/test/*.cc)
TESTS_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_SRC:%=%.o))
TESTS_DEP := $(TESTS_OBJ:$.o=$.d)
TESTS_CUSRC := $(wildcard $(TARGET_DIR)/test/*.cu)
TESTS_CUOBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_CUSRC:%=%.o))
TESTS_CUDADLINK := $(TESTS_CUOBJ:$cu.o=$cudadlink.o)
TESTS_CUDEP := $(TESTS_CUOBJ:$.o=$.d)
TESTS_EXE_CPU := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%,$(TESTS_SRC))
TESTS_EXE_CUDA :=  $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/%.cu,$(TEST_DIR)/$(TARGET_NAME)/%,$(TESTS_CUSRC))
TESTS_EXE := $(TESTS_EXE_CPU) $(TESTS_EXE_CUDA)
ALL_DEPENDS += $(TESTS_DEP) $(TESTS_CUDEP)
# Needed to keep the unit test object files after building $(TARGET)
.SECONDARY: $(TESTS_OBJ) $(TESTS_CUOBJ) $(TESTS_CUDADLINK)

define RUNTEST_template
run_$(1): $(1)
	@echo
	@echo "Running test $(1)"
	@$(1)
	@echo "Succeeded"
test_$(2): run_$(1)
endef
$(foreach test,$(TESTS_EXE_CPU),$(eval $(call RUNTEST_template,$(test),cpu)))
TEST_EXE_CUDA_RUN := $(filter-out $(TEST_DIR)/$(TARGET_NAME)/testEigenGPUNoFit,$(TESTS_EXE_CUDA))
$(foreach test,$(TEST_EXE_CUDA_RUN),$(eval $(call RUNTEST_template,$(test),nvidiagpu)))

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

$(OBJ_DIR)/$(2)/%.cu.o: $(SRC_DIR)/$(2)/%.cu
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CUDA_NVCC) $(CUDA_CUFLAGS) $(CUDA_CXXFLAGS) $(MY_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_NVCC_CXXFLAGS)) -c $$< -o $$@ -MMD

$$($(1)_CUDADLINK): $$($(1)_CUOBJ)
	$(CUDA_NVCC) $(CUDA_DLINKFLAGS) $(CUDA_LDFLAGS) $$($(1)_CUOBJ) -o $$@

$$($(1)_LIB): $$($(1)_OBJ) $$($(1)_CUDADLINK) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(CXX) $$($(1)_OBJ) $$($(1)_CUOBJ) $$($(1)_CUDADLINK) $(LDFLAGS) -shared $(SO_LDFLAGS) $(LIB_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) -o $$@
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

# Tests
$(OBJ_DIR)/$(TARGET_NAME)/test/%.cc.o: $(SRC_DIR)/$(TARGET_NAME)/test/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CUDA_TEST_CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD
	@cp $(@D)/$*.cc.d $(@D)/$*.cc.d.tmp; \
	  sed 's#\($(TARGET_NAME)/$*\)\.o[ :]*#\1.o \1.d : #g' < $(@D)/$*.cc.d.tmp > $(@D)/$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' -e 's/ *//' < $(@D)/$*.cc.d.tmp >> $(@D)/$*.cc.d; \
	  rm $(@D)/$*.cc.d.tmp

$(TEST_DIR)/$(TARGET_NAME)/%: $(OBJ_DIR)/$(TARGET_NAME)/test/%.cc.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $< $(LDFLAGS) $(MY_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

$(OBJ_DIR)/$(TARGET_NAME)/test/%.cu.o: $(SRC_DIR)/$(TARGET_NAME)/test/%.cu
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CUDA_NVCC) $(CUDA_CUFLAGS) $(CUDA_CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_NVCC_CXXFLAGS)) -c $< -o $@ -MMD

$(OBJ_DIR)/$(TARGET_NAME)/test/%.cudadlink.o: $(OBJ_DIR)/$(TARGET_NAME)/test/%.cu.o
	$(CUDA_NVCC) $(CUDA_DLINKFLAGS) $(CUDA_LDFLAGS) $< -o $@

$(TEST_DIR)/$(TARGET_NAME)/%: $(OBJ_DIR)/$(TARGET_NAME)/test/%.cu.o $(OBJ_DIR)/$(TARGET_NAME)/test/%.cudadlink.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) $(MY_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))
