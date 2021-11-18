#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "AlpakaCore/alpakaCommon.h"
#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "Framework/ESPluginFactory.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelGainCalibrationForHLTESProducer : public edm::ESProducer {
  public:
    explicit SiPixelGainCalibrationForHLTESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
    void produce(edm::EventSetup& eventSetup);

  private:
    std::filesystem::path data_;
  };

  void SiPixelGainCalibrationForHLTESProducer::produce(edm::EventSetup& eventSetup) {
    using DecodingStructure = SiPixelGainForHLTonGPU::DecodingStructure;

    std::ifstream in((data_ / "gain.bin"), std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    SiPixelGainForHLTonGPU gain;
    in.read(reinterpret_cast<char*>(&gain), sizeof(SiPixelGainForHLTonGPU));
    unsigned int nbytes;
    in.read(reinterpret_cast<char*>(&nbytes), sizeof(unsigned int));
    std::vector<DecodingStructure> gainData(nbytes / sizeof(DecodingStructure));
    in.read(reinterpret_cast<char*>(gainData.data()), nbytes);
    eventSetup.put(std::make_unique<SiPixelGainCalibrationForHLTGPU>(gain, std::move(gainData)));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(SiPixelGainCalibrationForHLTESProducer);
