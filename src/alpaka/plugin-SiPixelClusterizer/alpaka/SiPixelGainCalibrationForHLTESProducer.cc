#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include "AlpakaCore/device_unique_ptr.h"

#include <fstream>
#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct SiPixelGainForHLTonGPUBinary {
    using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;
    using Fields = SiPixelGainForHLTonGPU_Fields;

    using Range = std::pair<uint32_t, uint32_t>;
    using RangeAndCols = std::pair<Range, int>;

    DecodingStructure* v_pedestals;
    RangeAndCols rangeAndCols[2000];

    Fields fields_;
  };

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

    // TODO FIXME use the correct device
    Queue queue(devices[0]);

    const uint32_t numDecodingStructures = gainData.size() / sizeof(SiPixelGainForHLTonGPU_DecodingStructure);
    auto ped_h{cms::alpakatools::createHostView<SiPixelGainForHLTonGPU::DecodingStructure>(
        reinterpret_cast<SiPixelGainForHLTonGPU::DecodingStructure*>(gainData.data()), numDecodingStructures)};
    auto ped_d{cms::alpakatools::make_device_unique<SiPixelGainForHLTonGPU::DecodingStructure>(numDecodingStructures)};
    auto ped_d_view = cms::alpakatools::createDeviceView<SiPixelGainForHLTonGPU::DecodingStructure>(
        alpaka::getDev(queue), ped_d.get(), numDecodingStructures);
    alpaka::memcpy(queue, ped_d_view, ped_h, numDecodingStructures);

    auto rangeAndCols_h{
        cms::alpakatools::createHostView<SiPixelGainForHLTonGPU::RangeAndCols>(gain.rangeAndCols, 2000u)};
    auto rangeAndCols_d{cms::alpakatools::make_device_unique<SiPixelGainForHLTonGPU::RangeAndCols>(2000u)};
    auto rangeAndCols_d_view = cms::alpakatools::createDeviceView<SiPixelGainForHLTonGPU::RangeAndCols>(
        alpaka::getDev(queue), rangeAndCols_d.get(), 2000u);
    alpaka::memcpy(queue, rangeAndCols_d_view, rangeAndCols_h, 2000u);

    auto fields_h{cms::alpakatools::createHostView<SiPixelGainForHLTonGPU::Fields>(&gain.fields_, 1u)};
    auto fields_d{cms::alpakatools::make_device_unique<SiPixelGainForHLTonGPU::Fields>(1u)};
    auto fields_d_view =
        cms::alpakatools::createDeviceView<SiPixelGainForHLTonGPU::Fields>(alpaka::getDev(queue), fields_d.get(), 1u);
    alpaka::memcpy(queue, fields_d_view, fields_h, 1u);

    alpaka::wait(queue);

    eventSetup.put(
        std::make_unique<SiPixelGainForHLTonGPU>(std::move(ped_d), std::move(rangeAndCols_d), std::move(fields_d)));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(SiPixelGainCalibrationForHLTESProducer);
