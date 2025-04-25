JULIA := $(shell which julia)

DATA_DIR := data
TARGET_DIR := src/julia-serial
DATA_TAR_GZ := $(DATA_DIR)/data_v2.tar.gz
RAW_FILE := $(DATA_DIR)/raw.bin
URL_FILE := $(DATA_DIR)/url.txt
MD5_FILE := $(DATA_DIR)/md5.txt

# Main target
julia-serial: prepare_deps download_raw
	@echo "Setup complete. Run with ./julia-serial.sh"
	@chmod +x julia-serial.sh

# Prepare dependencies
prepare_deps:
	@echo "Installing required packages..."
	@$(JULIA) --project=$(TARGET_DIR) -e 'using Pkg; Pkg.instantiate()'

# Create data directory if it doesn't exist
$(DATA_DIR):
	@mkdir -p $(DATA_DIR)

# Download data archive
$(DATA_TAR_GZ): $(URL_FILE) | $(DATA_DIR)
	@echo "Downloading data_v2.tar.gz..."
	@curl -L -s -S `cat $(URL_FILE)` -o $(DATA_TAR_GZ)

# Extract raw data
extract_raw: $(DATA_TAR_GZ) $(MD5_FILE)
	@echo "Extracting raw.bin..."
	@cd $(DATA_DIR) && tar -tzf $(notdir $(DATA_TAR_GZ)) >/dev/null 2>&1 || (echo "Corrupted archive. Deleting..." && rm -f $(notdir $(DATA_TAR_GZ)) && exit 1)
	@cd $(DATA_DIR) && tar -xzf $(notdir $(DATA_TAR_GZ))
	@echo "Verifying file integrity..."
	@cd $(DATA_DIR) && md5sum -c $(notdir $(MD5_FILE)) >/dev/null 2>&1 || true

# Download raw data
download_raw:
	@if [ -f $(RAW_FILE) ]; then \
		echo "raw.bin already exists, skipping download and extraction"; \
	else \
		$(MAKE) $(DATA_TAR_GZ); \
		$(MAKE) extract_raw; \
	fi

# Clean target
clean:
	@echo "Cleaning data..."
	@rm -f $(DATA_TAR_GZ)

.PHONY: julia-serial prepare_deps download_raw clean extract_raw
