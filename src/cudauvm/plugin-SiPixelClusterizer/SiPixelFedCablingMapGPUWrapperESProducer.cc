#include "CondFormats/SiPixelFedIds.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include <fstream>
#include <memory>

class SiPixelFedCablingMapGPUWrapperESProducer : public edm::ESProducer {
public:
  explicit SiPixelFedCablingMapGPUWrapperESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void SiPixelFedCablingMapGPUWrapperESProducer::produce(edm::EventSetup& eventSetup) {
  {
    std::ifstream in(data_ / "fedIds.bin", std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    unsigned int nfeds;
    in.read(reinterpret_cast<char*>(&nfeds), sizeof(unsigned));
    std::vector<unsigned int> fedIds(nfeds);
    in.read(reinterpret_cast<char*>(fedIds.data()), sizeof(unsigned int) * nfeds);
    eventSetup.put(std::make_unique<SiPixelFedIds>(std::move(fedIds)));
  }
  {
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
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelFedCablingMapGPUWrapperESProducer);
