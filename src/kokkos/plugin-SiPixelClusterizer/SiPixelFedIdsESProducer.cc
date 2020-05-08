#include "CondFormats/SiPixelFedIds.h"

#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include <filesystem>
#include <fstream>
#include <memory>

class SiPixelFedIdsESProducer : public edm::ESProducer {
public:
  explicit SiPixelFedIdsESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void SiPixelFedIdsESProducer::produce(edm::EventSetup& eventSetup) {
  std::ifstream in(data_ / "fedIds.bin", std::ios::binary);
  in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
  unsigned int nfeds;
  in.read(reinterpret_cast<char*>(&nfeds), sizeof(unsigned));
  std::vector<unsigned int> fedIds(nfeds);
  in.read(reinterpret_cast<char*>(fedIds.data()), sizeof(unsigned int) * nfeds);
  eventSetup.put(std::make_unique<SiPixelFedIds>(std::move(fedIds)));
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelFedIdsESProducer);
