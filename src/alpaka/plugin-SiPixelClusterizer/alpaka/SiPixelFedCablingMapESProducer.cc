#include "CondFormats/SiPixelFedCablingMapGPU.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include "AlpakaCore/device_unique_ptr.h"

#include <fstream>
#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class SiPixelFedCablingMapESProducer : public edm::ESProducer {
  public:
    explicit SiPixelFedCablingMapESProducer(std::string const& datadir) : data_(datadir) {}
    void produce(edm::EventSetup& eventSetup);

  private:
    std::string data_;
  };

  void SiPixelFedCablingMapESProducer::produce(edm::EventSetup& eventSetup) {
    std::ifstream in((data_ + "/cablingMap.bin").c_str(), std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    SiPixelFedCablingMapGPU obj;
    in.read(reinterpret_cast<char*>(&obj), sizeof(SiPixelFedCablingMapGPU));
    unsigned int modToUnpDefSize;
    in.read(reinterpret_cast<char*>(&modToUnpDefSize), sizeof(unsigned int));
    std::vector<unsigned char> modToUnpDefault(modToUnpDefSize);
    in.read(reinterpret_cast<char*>(modToUnpDefault.data()), modToUnpDefSize);

    Queue queue(device);

    auto cablingMap_h{cms::alpakatools::createHostView<SiPixelFedCablingMapGPU>(&obj, 1u)};
    auto cablingMap_d{cms::alpakatools::make_device_unique<SiPixelFedCablingMapGPU>(1u)};
    auto cablingMap_d_view = cms::alpakatools::createDeviceView<SiPixelFedCablingMapGPU>(cablingMap_d.get(), 1u);
    alpaka::memcpy(queue, cablingMap_d_view, cablingMap_h, 1u);
    eventSetup.put(std::make_unique<SiPixelFedCablingMapGPUWrapper>(std::move(cablingMap_d), true));

    auto modToUnp_h{cms::alpakatools::createHostView<unsigned char>(modToUnpDefault.data(), modToUnpDefSize)};
    auto modToUnp_d{cms::alpakatools::make_device_unique<unsigned char>(modToUnpDefSize)};
    auto modToUnp_d_view = cms::alpakatools::createDeviceView<unsigned char>(modToUnp_d.get(), modToUnpDefSize);
    alpaka::memcpy(queue, modToUnp_d_view, modToUnp_h, modToUnpDefSize);

    alpaka::wait(queue);

    eventSetup.put(std::make_unique<cms::alpakatools::device::unique_ptr<unsigned char>>(std::move(modToUnp_d)));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(SiPixelFedCablingMapESProducer);
