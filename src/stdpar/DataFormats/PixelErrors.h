#ifndef DataFormats_SiPixelDigi_interface_PixelErrors_h
#define DataFormats_SiPixelDigi_interface_PixelErrors_h

#include <map>
#include <vector>
#include <cstdint>

#include "DataFormats/SiPixelRawDataError.h"

// Better ideas for the placement of these?

struct PixelErrorCompact {
  uint32_t rawId {0};
  uint32_t word {0};
  uint8_t errorType {0};
  uint8_t fedId {0};
};

using PixelFormatterErrors = std::map<uint32_t, std::vector<SiPixelRawDataError>>;

#endif  // DataFormats_SiPixelDigi_interface_PixelErrors_h
