TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_NAME ?= $(notdir $(TARGET_DIR))
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(mojo-serial_EXTERNAL_DEPENDS)

MOJO := MODULAR_HOME=$(MOJO_BASE)/share/max $(MOJO_BASE)/bin/mojo
MOJO_FLAGS := -D ASSERT=none -O3 -g0 --march=native
MOJO_SRC := $(shell find $(TARGET_DIR)/MojoSerial -name '*.mojo')

# mojo-serial target
$(TARGET): $(MOJO_SRC)
	cd $(TARGET_DIR) && $(MOJO) build $(MOJO_FLAGS) -I . MojoSerial/bin/main.mojo -o $@

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
