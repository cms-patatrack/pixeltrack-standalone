#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include "KokkosCore/kokkosConfig.h"

#include <fstream>
#include <memory>

namespace {
  struct SiPixelGainForHLTonGPUBinary {
    using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;
    using Range = std::pair<uint32_t, uint32_t>;

    DecodingStructure* v_pedestals;
    std::pair<Range, int> rangeAndCols[2000];

    float minPed_, maxPed_, minGain_, maxGain_;

    float pedPrecision, gainPrecision;

    unsigned int numberOfRowsAveragedOver_;  // this is 80!!!!
    unsigned int nBinsToUseForEncoding_;
    unsigned int deadFlag_;
    unsigned int noisyFlag_;
  };
}  // namespace

namespace KOKKOS_NAMESPACE {
  class SiPixelGainCalibrationForHLTESProducer : public edm::ESProducer {
  public:
    explicit SiPixelGainCalibrationForHLTESProducer(std::string const& datadir) : data_(datadir) {}
    void produce(edm::EventSetup& eventSetup);

  private:
    std::string data_;
  };

  void SiPixelGainCalibrationForHLTESProducer::produce(edm::EventSetup& eventSetup) {
    std::ifstream in((data_ + "/gain.bin").c_str(), std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    SiPixelGainForHLTonGPUBinary gain;
    in.read(reinterpret_cast<char*>(&gain), sizeof(SiPixelGainForHLTonGPUBinary));
    unsigned int nbytes;
    in.read(reinterpret_cast<char*>(&nbytes), sizeof(unsigned int));
    std::vector<char> gainData(nbytes);
    in.read(gainData.data(), nbytes);

    typename SiPixelGainForHLTonGPU<KokkosExecSpace>::DecodingStructureWritableView ped_d(
        "ped_d", gainData.size() / sizeof(SiPixelGainForHLTonGPU_DecodingStructure));
    auto ped_h = Kokkos::create_mirror_view(ped_d);
    memcpy(ped_h.data(), gainData.data(), gainData.size());
    Kokkos::deep_copy(KokkosExecSpace(), ped_d, ped_h);

    typename SiPixelGainForHLTonGPU<KokkosExecSpace>::RangeAndColsWritableView rangeAndCols_d("rangeAndCols_d");
    auto rangeAndCols_h = Kokkos::create_mirror_view(rangeAndCols_d);
    for (size_t i = 0; i < 2000; ++i) {
      rangeAndCols_h[i] =
          Kokkos::make_pair(Kokkos::make_pair(gain.rangeAndCols[i].first.first, gain.rangeAndCols[i].first.second),
                            gain.rangeAndCols[i].second);
    }
    Kokkos::deep_copy(KokkosExecSpace(), rangeAndCols_d, rangeAndCols_h);

    typename SiPixelGainForHLTonGPU<KokkosExecSpace>::FieldsWritableView fields_d("fields_d");
    auto fields_h = Kokkos::create_mirror_view(fields_d);
#define COPY(name) fields_h.data()->name = gain.name
    COPY(minPed_);
    COPY(maxPed_);
    COPY(minGain_);
    COPY(maxGain_);
    COPY(pedPrecision);
    COPY(gainPrecision);
    COPY(numberOfRowsAveragedOver_);
    COPY(nBinsToUseForEncoding_);
    COPY(deadFlag_);
    COPY(noisyFlag_);
#undef COPY
    Kokkos::deep_copy(KokkosExecSpace(), fields_d, fields_h);

    eventSetup.put(std::make_unique<SiPixelGainForHLTonGPU<KokkosExecSpace>>(
        std::move(ped_d), std::move(rangeAndCols_d), std::move(fields_d)));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_EVENTSETUP_MODULE(SiPixelGainCalibrationForHLTESProducer);
