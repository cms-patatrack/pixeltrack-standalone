#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include "AlpakaCore/alpakaCommon.h"

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

    Queue queue(device);

    const uint32_t numDecodingStructures = gainData.size() / sizeof(SiPixelGainForHLTonGPU_DecodingStructure);
    auto ped_h{cms::alpakatools::createHostView<SiPixelGainForHLTonGPU::DecodingStructure>(host, reinterpret_cast<SiPixelGainForHLTonGPU::DecodingStructure*>(gainData.data()), numDecodingStructures)};
    auto ped_d{cms::alpakatools::allocDeviceBuf<SiPixelGainForHLTonGPU::DecodingStructure>(device, numDecodingStructures)};
    cms::alpakatools::memcpy(queue, ped_d, ped_h, numDecodingStructures);

    auto rangeAndCols_h{cms::alpakatools::createHostView<SiPixelGainForHLTonGPU::RangeAndCols>(host, gain.rangeAndCols, 2000u)};
    auto rangeAndCols_d{cms::alpakatools::allocDeviceBuf<SiPixelGainForHLTonGPU::RangeAndCols>(device, 2000u)};
    cms::alpakatools::memcpy(queue, rangeAndCols_d, rangeAndCols_h, 2000u);


    auto fields_h{cms::alpakatools::createHostView<SiPixelGainForHLTonGPU::Fields>(host, &gain.fields_)};
    auto fields_d{cms::alpakatools::allocDeviceBuf<SiPixelGainForHLTonGPU::Fields>(device)};
    cms::alpakatools::memcpy(queue, fields_d, fields_h);

    eventSetup.put(std::make_unique<SiPixelGainForHLTonGPU>(std::move(ped_d), std::move(rangeAndCols_d), std::move(fields_d)));

    alpaka::wait(queue);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(SiPixelGainCalibrationForHLTESProducer);
