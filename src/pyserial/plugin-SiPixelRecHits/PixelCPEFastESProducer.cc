#include "CondFormats/PixelCPEFast.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include <fstream>
#include <iostream>
#include <memory>

class PixelCPEFastESProducer : public edm::ESProducer {
public:
  explicit PixelCPEFastESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void PixelCPEFastESProducer::produce(edm::EventSetup& eventSetup) {
  eventSetup.put(std::make_unique<PixelCPEFast>(data_ / "cpefast.bin"));
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEFastESProducer);
