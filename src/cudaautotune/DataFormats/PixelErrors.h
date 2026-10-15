#ifndef DataFormats_SiPixelDigi_interface_PixelErrors_h
#define DataFormats_SiPixelDigi_interface_PixelErrors_h

#include <map>
#include <vector>
#include <cstdint>

#include "DataFormats/SiPixelRawDataError.h"

// Better ideas for the placement of these?

struct PixelErrorCompact {
  uint32_t rawId;
  uint32_t word;
  uint8_t errorType;
  uint8_t fedId;
};

using PixelFormatterErrors = std::map<uint32_t, std::vector<SiPixelRawDataError>>;

#endif  // DataFormats_SiPixelDigi_interface_PixelErrors_h
