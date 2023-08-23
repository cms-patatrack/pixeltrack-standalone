#include <memory>
#include <string>

#include "AlpakaCore/config.h"
#include "CondFormats/alpaka/PixelCPEFast.h"
#include "Framework/ESPluginFactory.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PixelCPEFastESProducer : public edm::ESProducer {
  public:
    explicit PixelCPEFastESProducer(std::string const &datadir) : data_(datadir) {}
    void produce(edm::EventSetup &eventSetup);

  private:
    std::string data_;
  };

  void PixelCPEFastESProducer::produce(edm::EventSetup &eventSetup) {
    eventSetup.put(std::make_unique<PixelCPEFast>((data_ + "/cpefast.bin").c_str()));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(PixelCPEFastESProducer);
