#include "DataFormats/SiPixelDigisSoA.h"

#include <cassert>
#include <alpaka/alpaka.hpp>

SiPixelDigisSoA::SiPixelDigisSoA(
    size_t nDigis, const uint32_t *pdigi, const uint32_t *rawIdArr, const uint16_t *adc, const int32_t *clus)
    : pdigi_(pdigi, pdigi + nDigis),
      rawIdArr_(rawIdArr, rawIdArr + nDigis),
      adc_(adc, adc + nDigis),
      clus_(clus, clus + nDigis) {
  ALPAKA_ASSERT_OFFLOAD(pdigi_.size() == nDigis);
}
