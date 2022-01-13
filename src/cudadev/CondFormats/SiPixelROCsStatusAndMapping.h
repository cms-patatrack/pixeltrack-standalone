#ifndef CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h
#define CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h

#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

namespace pixelgpudetails {
  // Maximum fed for phase1 is 150 but not all of them are filled
  // Update the number FED based on maximum fed found in the cabling map
  constexpr unsigned int MAX_FED = 150;
  constexpr unsigned int MAX_LINK = 48;  // maximum links/channels for Phase 1
  constexpr unsigned int MAX_ROC = 8;
  constexpr unsigned int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;
  constexpr unsigned int MAX_SIZE_BYTE_BOOL = MAX_SIZE * sizeof(unsigned char);
}  // namespace pixelgpudetails

struct SiPixelROCsStatusAndMapping {
  alignas(128) unsigned int fed[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int link[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int roc[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int rawId[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int rocInDet[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int moduleId[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned char badRocs[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int size = 0;
};

GENERATE_SOA_LAYOUT(SiPixelROCsStatusAndMappingLayoutTemplate,
  SOA_COLUMN(unsigned int, fed),
  SOA_COLUMN(unsigned int, link),
  SOA_COLUMN(unsigned int, roc),
  SOA_COLUMN(unsigned int, rawId),
  SOA_COLUMN(unsigned int, rocInDet),
  SOA_COLUMN(unsigned int, moduleId),
  SOA_COLUMN(unsigned char, badRocs),
  SOA_SCALAR(unsigned int, size)
)

using SiPixelROCsStatusAndMappingLayout = SiPixelROCsStatusAndMappingLayoutTemplate<>;

GENERATE_SOA_CONST_VIEW(SiPixelROCsStatusAndMappingConstViewTemplate,
  SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(SiPixelROCsStatusAndMappingLayout, mappingLayout)),
  SOA_VIEW_VALUE_LIST(
    SOA_VIEW_VALUE(mappingLayout, fed),
    SOA_VIEW_VALUE(mappingLayout, link),
    SOA_VIEW_VALUE(mappingLayout, roc),
    SOA_VIEW_VALUE(mappingLayout, rawId),
    SOA_VIEW_VALUE(mappingLayout, rocInDet),
    SOA_VIEW_VALUE(mappingLayout, moduleId),
    SOA_VIEW_VALUE(mappingLayout, badRocs),
    SOA_VIEW_VALUE(mappingLayout, size)
  )
)

// Slightly more complex than using, but allows forward declarations.
struct SiPixelROCsStatusAndMappingConstView: public SiPixelROCsStatusAndMappingConstViewTemplate<> { 
  using SiPixelROCsStatusAndMappingConstViewTemplate<>::SiPixelROCsStatusAndMappingConstViewTemplate;
};

#endif  // CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h
