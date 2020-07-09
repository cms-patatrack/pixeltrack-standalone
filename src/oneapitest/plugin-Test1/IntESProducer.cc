#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

class IntESProducer : public edm::ESProducer {
public:
  IntESProducer(std::filesystem::path const& datadir){};

private:
  void produce(edm::EventSetup& eventSetup) { eventSetup.put(std::make_unique<int>(42)); }
};

DEFINE_FWK_EVENTSETUP_MODULE(IntESProducer);
