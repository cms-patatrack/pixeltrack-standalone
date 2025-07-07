#ifndef CondFormats_alpaka_SiPixelGainForHLTonGPU_h
#define CondFormats_alpaka_SiPixelGainForHLTonGPU_h

#include <utility>

#include "AlpakaCore/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelGainForHLTonGPU {
  public:
    struct DecodingStructure {
      uint8_t gain;
      uint8_t ped;
    };

    using Range = std::pair<uint32_t, uint32_t>;
    using RangeAndCols = std::pair<Range, int>;

    ALPAKA_FN_INLINE ALPAKA_FN_ACC std::pair<float, float> getPedAndGain(
        uint32_t moduleInd, int col, int row, bool& isDeadColumn, bool& isNoisyColumn) const {
      auto range = rangeAndCols_[moduleInd].first;
      auto nCols = rangeAndCols_[moduleInd].second;

      // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
      unsigned int lengthOfColumnData = (range.second - range.first) / nCols;
      unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block
      unsigned int numberOfDataBlocksToSkip = row / numberOfRowsAveragedOver_;

      auto offset =
          range.first + col * lengthOfColumnData + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip;

      ALPAKA_ASSERT_ACC(offset < range.second);
      ALPAKA_ASSERT_ACC(offset < 3088384);
      ALPAKA_ASSERT_ACC(0 == offset % 2);

      DecodingStructure const* __restrict__ lp = pedestals_;
      auto s = lp[offset / 2];
      isDeadColumn = (s.ped & 0xFF) == deadFlag_;
      isNoisyColumn = (s.ped & 0xFF) == noisyFlag_;

      return std::make_pair(decodePed(s.ped & 0xFF), decodeGain(s.gain & 0xFF));
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC float decodeGain(unsigned int gain) const {
      return gain * gainPrecision_ + minGain_;
    }
    ALPAKA_FN_INLINE ALPAKA_FN_ACC float decodePed(unsigned int ped) const { return ped * pedPrecision_ + minPed_; }

    DecodingStructure* pedestals_;
    RangeAndCols rangeAndCols_[2000];
    float minPed_, maxPed_, minGain_, maxGain_;
    float pedPrecision_, gainPrecision_;
    unsigned int numberOfRowsAveragedOver_;  // this is 80!!!!
    unsigned int nBinsToUseForEncoding_;
    unsigned int deadFlag_;
    unsigned int noisyFlag_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_alpaka_SiPixelGainForHLTonGPU_h
