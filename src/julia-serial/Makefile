TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
#TARGET_NAME is set in the top-level Makefile
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(alpaka_EXTERNAL_DEPENDS)

# julia-serial target
$(TARGET): julia-serial-build
	@echo "Setup complete. Run with ./$(TARGET_NAME)"
	@echo -e '#! /bin/bash\njulia --project=$(TARGET_DIR) $(TARGET_DIR)/run_main.jl "$$@"' > $@
	@chmod +x $@

# Prepare dependencies
.PHONY: julia-serial-build
julia-serial-build:
	@echo "Installing required packages..."
	@julia --project=$(TARGET_DIR) -e 'using Pkg; Pkg.instantiate()'
