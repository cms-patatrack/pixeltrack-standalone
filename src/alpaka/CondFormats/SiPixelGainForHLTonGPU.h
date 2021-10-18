#ifndef CondFormats_SiPixelObjects_SiPixelGainForHLTonGPU_h
#define CondFormats_SiPixelObjects_SiPixelGainForHLTonGPU_h

#include "AlpakaCore/device_unique_ptr.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct SiPixelGainForHLTonGPU_DecodingStructure {
    uint8_t gain;
    uint8_t ped;
  };

  struct SiPixelGainForHLTonGPU_Fields {
    float minPed_, maxPed_, minGain_, maxGain_;

    float pedPrecision, gainPrecision;

    unsigned int numberOfRowsAveragedOver_;  // this is 80!!!!
    unsigned int nBinsToUseForEncoding_;
    unsigned int deadFlag_;
    unsigned int noisyFlag_;
  };

  class SiPixelGainForHLTonGPU {
  public:
    using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;
    using Fields = SiPixelGainForHLTonGPU_Fields;

    using Range = std::pair<uint32_t, uint32_t>;
    using RangeAndCols = std::pair<Range, int>;

    SiPixelGainForHLTonGPU(cms::alpakatools::device::unique_ptr<DecodingStructure> ped,
                           cms::alpakatools::device::unique_ptr<RangeAndCols> rc,
                           cms::alpakatools::device::unique_ptr<Fields> f)
        : v_pedestals_(std::move(ped)), rangeAndCols_(std::move(rc)), fields_(std::move(f)){};

    ALPAKA_FN_INLINE ALPAKA_FN_ACC static std::pair<float, float> getPedAndGain(const DecodingStructure* v_pedestals,
                                                                                const RangeAndCols* rangeAndCols,
                                                                                const Fields* fields,
                                                                                uint32_t moduleInd,
                                                                                int col,
                                                                                int row,
                                                                                bool& isDeadColumn,
                                                                                bool& isNoisyColumn) {
      auto range = rangeAndCols[moduleInd].first;
      auto nCols = rangeAndCols[moduleInd].second;

      // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
      unsigned int lengthOfColumnData = (range.second - range.first) / nCols;
      unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block
      unsigned int numberOfDataBlocksToSkip = row / fields->numberOfRowsAveragedOver_;

      auto offset =
          range.first + col * lengthOfColumnData + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip;

      assert(offset < range.second);
      assert(offset < 3088384);
      assert(0 == offset % 2);

      auto s = v_pedestals[offset / 2];

      isDeadColumn = (s.ped & 0xFF) == fields->deadFlag_;
      isNoisyColumn = (s.ped & 0xFF) == fields->noisyFlag_;

      return std::make_pair(decodePed(fields, s.ped & 0xFF), decodeGain(fields, s.gain & 0xFF));
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC static float decodeGain(const Fields* fields, unsigned int gain) {
      return gain * fields->gainPrecision + fields->minGain_;
    }
    ALPAKA_FN_INLINE ALPAKA_FN_ACC static float decodePed(const Fields* fields, unsigned int ped) {
      return ped * fields->pedPrecision + fields->minPed_;
    }

    ALPAKA_FN_HOST const DecodingStructure* getVpedestals() const { return v_pedestals_.get(); }
    ALPAKA_FN_HOST const RangeAndCols* getRangeAndCols() const { return rangeAndCols_.get(); }
    ALPAKA_FN_HOST const Fields* getFields() const { return fields_.get(); }

  private:
    cms::alpakatools::device::unique_ptr<DecodingStructure> v_pedestals_;
    cms::alpakatools::device::unique_ptr<RangeAndCols> rangeAndCols_;
    cms::alpakatools::device::unique_ptr<Fields> fields_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
