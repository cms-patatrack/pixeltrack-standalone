#ifndef CondFormats_SiPixelObjects_SiPixelGainForHLTonGPU_h
#define CondFormats_SiPixelObjects_SiPixelGainForHLTonGPU_h

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct SiPixelGainForHLTonGPU_DecodingStructure {
    uint8_t gain;
    uint8_t ped;
  };

  class SiPixelGainForHLTonGPU {
  public:
    using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;
    using Range = std::pair<uint32_t, uint32_t>;
    using RangeAndCols = std::pair<Range, int>;
      
    ALPAKA_FN_INLINE ALPAKA_FN_ACC std::pair<float, float> getPedAndGain(uint32_t moduleInd,
                                                                                int col,
                                                                                int row,
                                                                                bool& isDeadColumn,
                                                                                bool& isNoisyColumn) const {
      
      auto range = rangeAndCols[moduleInd].first;
      auto nCols = rangeAndCols[moduleInd].second;

      // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
      unsigned int lengthOfColumnData = (range.second - range.first) / nCols;
      unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block
      unsigned int numberOfDataBlocksToSkip = row / numberOfRowsAveragedOver_;

      auto offset =
          range.first + col * lengthOfColumnData + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip;

      ALPAKA_ASSERT_OFFLOAD(offset < range.second);
      ALPAKA_ASSERT_OFFLOAD(offset < 3088384);
      ALPAKA_ASSERT_OFFLOAD(0 == offset % 2);

      DecodingStructure const* __restrict__ lp = v_pedestals;
      auto s = lp[offset / 2];
      isDeadColumn = (s.ped & 0xFF) == deadFlag_;
      isNoisyColumn = (s.ped & 0xFF) == noisyFlag_;

      return std::make_pair(decodePed(s.ped & 0xFF), decodeGain(s.gain & 0xFF));
    }

    ALPAKA_FN_INLINE  ALPAKA_FN_ACC float decodeGain(unsigned int gain) const { return gain * gainPrecision + minGain_; }
    ALPAKA_FN_INLINE  ALPAKA_FN_ACC float decodePed(unsigned int ped) const { return ped * pedPrecision + minPed_; }

    DecodingStructure* v_pedestals;

    RangeAndCols rangeAndCols[2000];

    float minPed_, maxPed_, minGain_, maxGain_;

    float pedPrecision, gainPrecision;

    unsigned int numberOfRowsAveragedOver_;  // this is 80!!!!
    unsigned int nBinsToUseForEncoding_;
    unsigned int deadFlag_;
    unsigned int noisyFlag_;

  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
