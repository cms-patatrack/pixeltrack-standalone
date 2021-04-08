#ifndef CondFormats_SiPixelObjects_SiPixelGainForHLTonGPU_h
#define CondFormats_SiPixelObjects_SiPixelGainForHLTonGPU_h

#include "AlpakaCore/alpakaCommon.h"


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


  SiPixelGainForHLTonGPU(AlpakaDeviceBuf<DecodingStructure> ped, AlpakaDeviceBuf<RangeAndCols> rc, AlpakaDeviceBuf<Fields> f)
      : v_pedestals(std::move(ped)), rangeAndCols(std::move(rc)), fields(std::move(f)){};

  ALPAKA_FN_INLINE std::pair<float, float> getPedAndGain(
      uint32_t moduleInd, int col, int row, bool& isDeadColumn, bool& isNoisyColumn) const {
      auto range = getRangeAndCols()[moduleInd].first;
      auto nCols = getRangeAndCols()[moduleInd].second;

    // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
    unsigned int lengthOfColumnData = (range.second - range.first) / nCols;
    unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block
    unsigned int numberOfDataBlocksToSkip = row / getFields()->numberOfRowsAveragedOver_;

    auto offset = range.first + col * lengthOfColumnData + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip;

    assert(offset < range.second);
    assert(offset < 3088384);
    assert(0 == offset % 2);

   auto s = getVpedestals()[offset / 2];

    isDeadColumn = (s.ped & 0xFF) == getFields()->deadFlag_;
    isNoisyColumn = (s.ped & 0xFF) == getFields()->noisyFlag_;

    return std::make_pair(decodePed(s.ped & 0xFF), decodeGain(s.gain & 0xFF));
  }

  ALPAKA_FN_INLINE float decodeGain(unsigned int gain) const {
    return gain * getFields()->gainPrecision + getFields()->minGain_;
  }
  ALPAKA_FN_INLINE float decodePed(unsigned int ped) const {
    return ped * getFields()->pedPrecision + getFields()->minPed_;
  }

private:
  const DecodingStructure* getVpedestals() const { return alpaka::getPtrNative(v_pedestals); }
  const RangeAndCols* getRangeAndCols() const { return alpaka::getPtrNative(rangeAndCols); }
  const Fields* getFields() const { return alpaka::getPtrNative(fields); }


  AlpakaDeviceBuf<DecodingStructure> v_pedestals;
  AlpakaDeviceBuf<RangeAndCols> rangeAndCols;
  AlpakaDeviceBuf<Fields> fields;
};

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
