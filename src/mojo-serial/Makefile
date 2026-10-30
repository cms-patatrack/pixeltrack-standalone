TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_NAME ?= $(notdir $(TARGET_DIR))
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(mojo-serial_EXTERNAL_DEPENDS)

MOJO := MODULAR_HOME=$(MOJO_BASE)/share/max $(MOJO_BASE)/bin/mojo
# Do not fuse the multiply-adds across statements (the default is contract=fast), so that the results do not depend on how
# the code is written, like -ffp-contract=off for the C++ back-ends.
MOJO_FLAGS := -D ASSERT=none -O3 -g0 --march=native --fp-mode contract=off
MOJO_SRC := $(shell find $(TARGET_DIR)/MojoSerial -name '*.mojo')

# Plugins: each MojoSerial/plugin_<X> directory is built into the shared library
# $(LIB_DIR)/mojo-serial/plugin<X>.so, from its Plugin.mojo, and loaded at run
# time by bin/PluginManager.mojo. Mojo compiles the imported modules from their
# sources, so each library and the executable contain their own copy of the code
# they use (Framework, DataFormats, ...): there are no other libraries.
PLUGIN_DIR := $(LIB_DIR)/$(TARGET_NAME)
PLUGINNAMES := $(patsubst plugin_%,%,$(notdir $(wildcard $(TARGET_DIR)/MojoSerial/plugin_*)))
PLUGINS := $(patsubst %,$(PLUGIN_DIR)/plugin%.so,$(PLUGINNAMES))

# mojo-serial target
$(TARGET): $(MOJO_SRC) $(PLUGINS) $(PLUGIN_DIR)/plugins.txt
	cd $(TARGET_DIR) && $(MOJO) build $(MOJO_FLAGS) -D LIB_DIR=$(PLUGIN_DIR) -I . MojoSerial/bin/main.mojo -o $@

$(PLUGIN_DIR)/plugin%.so: $(MOJO_SRC)
	@[ -d $(@D) ] || mkdir -p $(@D)
	cd $(TARGET_DIR) && $(MOJO) build $(MOJO_FLAGS) --emit shared-lib -I . MojoSerial/plugin_$*/Plugin.mojo -o $@

# The modules of each library, from the functions it exports to register them.
$(PLUGIN_DIR)/plugins.txt: $(PLUGINS)
	nm -A -D -P --defined-only $(PLUGINS) | sed -n -e"s#$(PLUGIN_DIR)/\(plugin\w\+\.so\): edmplugin_E[DS]Producer_\(\w\+\) T .*#\2 \1#p" | sort > $@

test_cpu: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --data $(BASE_DIR)/data --maxEvents 2
	@echo "Succeeded"
test_nvidiagpu:
test_amdgpu:
test_intelgpu:
test_auto:
.PHONY: test_cpu test_nvidiagpu test_amdgpu test_intelgpu test_auto
