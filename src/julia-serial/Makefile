TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
#TARGET_NAME is set in the top-level Makefile
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(alpaka_EXTERNAL_DEPENDS)

# julia-serial target
$(TARGET): julia-serial-build bin/julia-serial
	@cat $(TARGET_DIR)/bin/julia-serial | sed -e"s#!TARGET_DIR#$(TARGET_DIR)#g" > $@
	@chmod +x $@
	@echo "Setup complete. Run with ./$(TARGET_NAME)"

# Prepare dependencies
.PHONY: julia-serial-build
julia-serial-build:
	@echo "Installing required packages..."
	@julia --project=$(TARGET_DIR) -e 'using Pkg; isfile("$(TARGET_DIR)/Manifest.toml") ? Pkg.resolve() : Pkg.instantiate()' 
