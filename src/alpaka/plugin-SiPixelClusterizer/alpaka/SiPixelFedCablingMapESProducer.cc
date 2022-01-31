#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "AlpakaCore/alpakaConfig.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"
#include "CondFormats/alpaka/SiPixelFedCablingMapGPUWrapper.h"
#include "Framework/ESPluginFactory.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class SiPixelFedCablingMapESProducer : public edm::ESProducer {
  public:
    explicit SiPixelFedCablingMapESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
    void produce(edm::EventSetup& eventSetup);

  private:
    std::filesystem::path data_;
  };

  void SiPixelFedCablingMapESProducer::produce(edm::EventSetup& eventSetup) {
    std::ifstream in(data_ / "cablingMap.bin", std::ios::binary);

    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    SiPixelFedCablingMapGPU obj;
    in.read(reinterpret_cast<char*>(&obj), sizeof(SiPixelFedCablingMapGPU));
    unsigned int modToUnpDefSize;
    in.read(reinterpret_cast<char*>(&modToUnpDefSize), sizeof(unsigned int));
    std::vector<unsigned char> modToUnpDefault(modToUnpDefSize);
    in.read(reinterpret_cast<char*>(modToUnpDefault.data()), modToUnpDefSize);
    eventSetup.put(std::make_unique<SiPixelFedCablingMapGPUWrapper>(obj, std::move(modToUnpDefault)));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(SiPixelFedCablingMapESProducer);
