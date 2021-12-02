#ifndef CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h
#define CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h

#include "DataFormats/SoAStore.h"
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

generate_SoA_store(SiPixelROCsStatusAndMappingStoreTemplate,
  SoA_column(unsigned int, fed),
  SoA_column(unsigned int, link),
  SoA_column(unsigned int, roc),
  SoA_column(unsigned int, rawId),
  SoA_column(unsigned int, rocInDet),
  SoA_column(unsigned int, moduleId),
  SoA_column(unsigned char, badRocs),
  SoA_scalar(unsigned int, size)
);

using SiPixelROCsStatusAndMappingStore = SiPixelROCsStatusAndMappingStoreTemplate<>;

generate_SoA_const_view(SiPixelROCsStatusAndMappingConstViewTemplate,
  SoA_view_store_list(SoA_view_store(SiPixelROCsStatusAndMappingStore, mappingStore)),
  SoA_view_value_list(
    SoA_view_value(mappingStore, fed),
    SoA_view_value(mappingStore, link),
    SoA_view_value(mappingStore, roc),
    SoA_view_value(mappingStore, rawId),
    SoA_view_value(mappingStore, rocInDet),
    SoA_view_value(mappingStore, moduleId),
    SoA_view_value(mappingStore, badRocs),
    SoA_view_value(mappingStore, size)
  )
);

// Slightly more complex than using, but allows forward declarations.
struct SiPixelROCsStatusAndMappingConstView: public SiPixelROCsStatusAndMappingConstViewTemplate<> { 
  using SiPixelROCsStatusAndMappingConstViewTemplate<>::SiPixelROCsStatusAndMappingConstViewTemplate;
};

#endif  // CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h
