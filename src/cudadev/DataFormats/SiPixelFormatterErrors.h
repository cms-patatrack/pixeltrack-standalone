#ifndef DataFormats_SiPixelRawData_interface_SiPixelFormatterErrors_h
#define DataFormats_SiPixelRawData_interface_SiPixelFormatterErrors_h

#include <map>
#include <vector>

#include "DataFormats/SiPixelRawDataError.h"

using SiPixelFormatterErrors = std::map<uint32_t, std::vector<SiPixelRawDataError>>;

#endif  // DataFormats_SiPixelRawData_interface_SiPixelFormatterErrors_h
