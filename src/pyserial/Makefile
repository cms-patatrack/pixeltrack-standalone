TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_NAME := $(notdir $(TARGET_DIR))
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(pyserial_EXTERNAL_DEPENDS)

# ------------------------------------------------------------------ Python
#
# The interpreter's own flags cannot be known until it has been bootstrapped,
# so they come from a generated makefile.  Including it here rather than in the
# top-level Makefile means building a non-Python backend never triggers the
# bootstrap.  $(PYTHON_DEPS) makes it exist before this make runs.
include $(BASE_DIR)/external/python-config.mk

PYTHON_DIR   := $(TARGET_DIR)/python
PYCACHE_DIR  := $(OBJ_DIR)/$(TARGET_NAME)/pycache

PY_DEFINES := -DEDM_PYTHON_HOME='"$(PY_HOME)"' \
              -DEDM_PYTHON_SITE='"$(PY_SITE)"' \
              -DEDM_PYTHON_DIR='"$(PYTHON_DIR)"' \
              -DEDM_PYTHON_PYCACHE='"$(PYCACHE_DIR)"'

REFLECT_PRODUCTS := std::vector<float>
REFLECT_INCLUDES := -I$(TARGET_DIR)

GEN_DIR        := $(OBJ_DIR)/$(TARGET_NAME)/generated
REFLECT_HEADER := $(TARGET_DIR)/DataFormats/Products.h
DICT_SOURCE    := $(GEN_DIR)/products_dict.cxx
DICT_DEPEND    := $(GEN_DIR)/products_dict.d
DICT_LIB       := $(GEN_DIR)/libproducts_dict.so
REFLECTED      := $(GEN_DIR)/ReflectedBindings.generated.cc

# NB_SHARED is nanobind's own switch for "the runtime is a shared library
# rather than a copy per extension": it turns its core API from hidden symbols
# into exported ones.  libnanobind.so is built with NB_BUILD as well, and is the
# only place the amalgamation is compiled; everything else links against it, so
# there is one type registry in the process however many extensions there are.
MY_CXXFLAGS := -I$(TARGET_DIR) -DLIB_DIR=$(LIB_DIR)/$(TARGET_NAME) $(PY_CXXFLAGS) $(PY_DEFINES) \
               -DNB_SHARED \
               -DEDM_NB_COMBINED='"$(NB_COMBINED)"' \
               -DEDM_REFLECTED_BINDINGS='"$(REFLECTED)"'
MY_LDFLAGS  := -Wl,-rpath,$(LIB_DIR)/$(TARGET_NAME) -L$(LIB_DIR)/$(TARGET_NAME) $(PY_LDFLAGS)

$(TARGET):
test_cpu: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --maxEvents 2
	@echo "Succeeded"
test_nvidiagpu:
test_amdgpu:
test_intelagpu:
test_auto:
.PHONY: test_cpu test_nvidiagpu test_intelgpu test_auto

EXE_SRC := $(wildcard $(TARGET_DIR)/bin/*.cc)
EXE_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(EXE_SRC:%=%.o))
EXE_DEP := $(EXE_OBJ:$.o=$.d)

# A library is a directory that holds C++ sources.  Selecting on the sources
# rather than on the directory name keeps python/ and tools/, and the .ini
# configurations, from being built into empty libraries and linked against.
LIBNAMES := $(filter-out plugin-% bin test bindings,$(sort $(patsubst %/,%,$(dir $(wildcard */*.cc)))))
PLUGINNAMES := $(patsubst plugin-%,%,$(filter plugin-%,$(wildcard *)))

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
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
ALL_DEPENDS += $$($(1)_DEP)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1).so
PLUGINS += $$($(1)_LIB)
endef
$(foreach lib,$(PLUGINNAMES),$(eval $(call PLUGIN_template,$(lib))))

# Files for unit tests
TESTS_SRC := $(wildcard $(TARGET_DIR)/test/*.cc)
TESTS_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_SRC:%=%.o))
TESTS_DEP := $(TESTS_OBJ:$.o=$.d)
TESTS_EXE := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%,$(TESTS_SRC))
ALL_DEPENDS += $(TESTS_DEP)
# Needed to keep the unit test object files after building $(TARGET)
.SECONDARY: $(TESTS_OBJ)

define RUNTEST_template
run_$(1): $(1)
	@echo
	@echo "Running test $(1)"
	@$(1)
	@echo "Succeeded"
test_cpu: run_$(1)
endef
$(foreach test,$(TESTS_EXE),$(eval $(call RUNTEST_template,$(test))))

-include $(ALL_DEPENDS)

# Build targets
$(LIB_DIR)/$(TARGET_NAME)/plugins.txt: $(PLUGINS)
	nm -A -C -D -P --defined-only $(PLUGINS) | sed -n -e"s#$(LIB_DIR)/$(TARGET_NAME)/\(plugin\w\+\.so\): typeinfo for edm::\(PluginFactory\|ESPluginFactory\)::impl::Maker<\([A-Za-z0-9_:]\+\)> V .* .*#\3 \1#p" | sort > $@

# edm_core is a Python extension, not a plugin and not a library: the
# interpreter imports it by name from the library directory.  It names every
# product type, so it links against the product libraries and sits above them --
# which is what lets a bound class have its constructors and destructor out of
# line.  Nothing under Framework/ names a product any more.
EDM_CORE_SRC := $(wildcard $(TARGET_DIR)/bindings/*.cc)
EDM_CORE_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(EDM_CORE_SRC:%=%.o))
EDM_CORE_DEP := $(EDM_CORE_OBJ:$.o=$.d)
EDM_CORE := $(LIB_DIR)/$(TARGET_NAME)/edm_core.so
ALL_DEPENDS += $(EDM_CORE_DEP)

$(OBJ_DIR)/$(TARGET_NAME)/bindings/%.cc.o: $(SRC_DIR)/$(TARGET_NAME)/bindings/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD

$(EDM_CORE): $(EDM_CORE_OBJ) $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $(EDM_CORE_OBJ) $(LDFLAGS) -shared $(SO_LDFLAGS) $(MY_LDFLAGS) -o $@ \
		$(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

$(TARGET): $(EXE_OBJ) $(LIBS) $(PLUGINS) $(EDM_CORE) $(LIB_DIR)/$(TARGET_NAME)/plugins.txt | $(TESTS_EXE)
	$(CXX) $(EXE_OBJ) $(LDFLAGS) $(MY_LDFLAGS) -o $@ $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

# The one object that compiles nanobind's amalgamation, and the only one built
# with NB_BUILD.
$(OBJ_DIR)/$(TARGET_NAME)/nanobind/%.cc.o: MY_CXXFLAGS += -DNB_BUILD

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
	$(CXX) $$($(1)_OBJ) $(LDFLAGS) -shared $(SO_LDFLAGS) $(MY_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) $(MY_LDFLAGS) -o $$@
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
	$(CXX) $(CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) -c $< -o $@ -MMD
	@cp $(@D)/$*.cc.d $(@D)/$*.cc.d.tmp; \
	  sed 's#\($(TARGET_NAME)/$*\)\.o[ :]*#\1.o \1.d : #g' < $(@D)/$*.cc.d.tmp > $(@D)/$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' -e 's/ *//' < $(@D)/$*.cc.d.tmp >> $(@D)/$*.cc.d; \
	  rm $(@D)/$*.cc.d.tmp

$(TEST_DIR)/$(TARGET_NAME)/%: $(OBJ_DIR)/$(TARGET_NAME)/test/%.cc.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CXX) $< $(LDFLAGS) $(MY_LDFLAGS) -o $@ $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))


# --------------------------------------------------- generated bindings
#
# rootcling parses the product header with cling and emits a dictionary; the
# dictionary is compiled into a shared library; a ROOT macro walks it through
# TClass and writes the nanobind registrations for every product and everything
# reachable from one.  No binding in this backend is written by hand.
#
# ROOT is a build-time dependency only: nothing links against it.
comma := ,
empty :=
space := $(empty) $(empty)
REFLECT_LIST := $(subst $(space),$(comma),$(strip $(REFLECT_PRODUCTS)))

# A scalar product crosses to Python through nanobind's own caster and has no
# class to reflect, so it is left out of the dictionary: rootcling has nothing
# to generate for a fundamental type and says so once per rule.  Only the
# single-word spellings are listed, because a space-separated make variable
# cannot hold `unsigned int` in the first place.
SCALAR_TYPES := bool char short int long float double size_t \
                int8_t uint8_t int16_t uint16_t int32_t uint32_t int64_t uint64_t
REFLECT_CLASSES := $(filter-out $(SCALAR_TYPES),$(REFLECT_PRODUCTS))

# ROOT_HIST=0 keeps the interpreter from dropping a .root_hist in $$HOME.
#
# EXTRA_CLING_ARGS is how a flag reaches cling: rootcling's own option parser
# reads -std as -s, and this ROOT was built for C++17, so its interpreter
# defaults to C++17 and would not know std::span.  CMSSW does not need this,
# because there ROOT is built with the same standard as everything else.
#
# LD_LIBRARY_PATH is there because the dictionary links against the product
# libraries, which find their own dependencies through it rather than through a
# runpath, the way everything in this repository does.  env.sh sets that up for
# running a job; the build has to set it up for itself, since ROOT is what
# loads the dictionary here.
EXTERNAL_LIBDIRS := $(patsubst -L%,%,$(filter -L%,$(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))))
ROOT_ENV := ROOT_HIST=0 EXTRA_CLING_ARGS=-std=c++20 \
            LD_LIBRARY_PATH=$(subst $(space),:,$(EXTERNAL_LIBDIRS)):$$LD_LIBRARY_PATH

$(GEN_DIR):
	@mkdir -p $@

$(GEN_DIR)/LinkDef.h: $(TARGET_DIR)/Makefile | $(GEN_DIR)
	@printf '%s\n' \
		'#ifdef __ROOTCLING__' \
		'#pragma link off all globals;' \
		'#pragma link off all classes;' \
		'#pragma link off all functions;' \
		$(foreach t,$(REFLECT_CLASSES),'#pragma link C++ class $(t)+;') \
		'#endif' > $@

# The dictionary has to be rebuilt when a product's *declaration* changes, not
# only when the header listing the products does.  Depending on the product
# header alone means an edit to a class it includes leaves a stale dictionary
# behind: the build succeeds and the bindings are silently the old ones, which
# a Python module then discovers at run time.  The header list comes from the
# compiler, the way it does for every other object here.
$(DICT_DEPEND): $(REFLECT_HEADER) | $(GEN_DIR)
	@$(CXX) $(CXXFLAGS) $(REFLECT_INCLUDES) -MM -MT '$(DICT_SOURCE)' $< > $@

-include $(DICT_DEPEND)

$(DICT_SOURCE): $(REFLECT_HEADER) $(GEN_DIR)/LinkDef.h | $(GEN_DIR)
	@echo "  ROOTCLING  $(notdir $(REFLECT_HEADER))"
	@cd $(GEN_DIR) && $(ROOT_ENV) rootcling -f $(notdir $@) \
		$(REFLECT_INCLUDES) DataFormats/Products.h LinkDef.h

# The dictionary is loaded by ROOT on its own, so it needs the symbols of the
# classes it describes -- a typeinfo, a vtable, whatever a member function it
# stubs refers to.  Linking it against the product libraries is what lets those
# be defined out of line, as they are everywhere else in this repository.
$(DICT_LIB): $(DICT_SOURCE) $(LIBS)
	@echo "  DICT       $(notdir $@)"
	@$(CXX) -O1 -shared -fPIC -o $@ $< $(REFLECT_INCLUDES) \
		$$($(ROOT_ENV) root-config --cflags --ldflags --libs) -std=c++20 \
		-Wl,-rpath,$(LIB_DIR)/$(TARGET_NAME) -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES))

$(REFLECTED): $(DICT_LIB) $(TARGET_DIR)/tools/generate_bindings.C
	@echo "  REFLECT    $(REFLECT_LIST)"
	@set -o pipefail; $(ROOT_ENV) root -l -b -q -n \
		'$(TARGET_DIR)/tools/generate_bindings.C("$(DICT_LIB)", "$@.tmp", "$(REFLECT_LIST)")' \
		| grep -v '^Processing\|^(int)' || { rm -f $@.tmp; exit 1; }
	@mv $@.tmp $@

# The wrapper translation unit #includes the generated file.
$(OBJ_DIR)/$(TARGET_NAME)/bindings/ReflectedBindings.cc.o: $(REFLECTED)
