#ifndef CondFormats_SiPixelObjects_SiPixelGainForHLTonGPU_h
#define CondFormats_SiPixelObjects_SiPixelGainForHLTonGPU_h

#include <Kokkos_Core.hpp>

struct SiPixelGainForHLTonGPU_DecodingStructure {
  uint8_t gain;
  uint8_t ped;
};

template <typename MemorySpace>
class SiPixelGainForHLTonGPU {
public:
  using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;
  struct Fields {
    float minPed_, maxPed_, minGain_, maxGain_;

    float pedPrecision, gainPrecision;

    unsigned int numberOfRowsAveragedOver_;  // this is 80!!!!
    unsigned int nBinsToUseForEncoding_;
    unsigned int deadFlag_;
    unsigned int noisyFlag_;
  };

  using Range = Kokkos::pair<uint32_t, uint32_t>;

  using DecodingStructureWritableView = Kokkos::View<DecodingStructure*, MemorySpace>;
  using RangeAndColsWritableView = Kokkos::View<Kokkos::pair<Range, int>[2000], MemorySpace>;
  using FieldsWritableView = Kokkos::View<Fields, MemorySpace>;

  using DecodingStructureView = Kokkos::View<DecodingStructure const*, MemorySpace>;
  using RangeAndColsView = Kokkos::View<Kokkos::pair<Range, int> const[2000], MemorySpace>;
  using FieldsView = Kokkos::View<Fields const, MemorySpace>;

  SiPixelGainForHLTonGPU(DecodingStructureView ped, RangeAndColsView rc, FieldsView f)
      : v_pedestals(std::move(ped)), rangeAndCols(std::move(rc)), fields(std::move(f)){};

  KOKKOS_INLINE_FUNCTION std::pair<float, float> getPedAndGain(
      uint32_t moduleInd, int col, int row, bool& isDeadColumn, bool& isNoisyColumn) const {
    auto range = rangeAndCols[moduleInd].first;
    auto nCols = rangeAndCols[moduleInd].second;

    // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
    unsigned int lengthOfColumnData = (range.second - range.first) / nCols;
    unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block
    unsigned int numberOfDataBlocksToSkip = row / fields.data()->numberOfRowsAveragedOver_;

    auto offset = range.first + col * lengthOfColumnData + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip;

    assert(offset < range.second);
    assert(offset < 3088384);
    assert(0 == offset % 2);

    auto s = v_pedestals[offset / 2];

    isDeadColumn = (s.ped & 0xFF) == fields.data()->deadFlag_;
    isNoisyColumn = (s.ped & 0xFF) == fields.data()->noisyFlag_;

    return std::make_pair(decodePed(s.ped & 0xFF), decodeGain(s.gain & 0xFF));
  }

  KOKKOS_INLINE_FUNCTION float decodeGain(unsigned int gain) const {
    return gain * fields.data()->gainPrecision + fields.data()->minGain_;
  }
  KOKKOS_INLINE_FUNCTION float decodePed(unsigned int ped) const {
    return ped * fields.data()->pedPrecision + fields.data()->minPed_;
  }

private:
  DecodingStructureView v_pedestals;
  RangeAndColsView rangeAndCols;
  FieldsView fields;
};

#endif
