#include "DataFormats/BeamSpotPOD.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include <fstream>
#include <iostream>
#include <memory>

class BeamSpotESProducer : public edm::ESProducer {
public:
  explicit BeamSpotESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void BeamSpotESProducer::produce(edm::EventSetup& eventSetup) {
  auto bs = std::make_unique<BeamSpotPOD>();

  std::ifstream in(data_ / "beamspot.bin", std::ios::binary);
  in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
  in.read(reinterpret_cast<char*>(bs.get()), sizeof(BeamSpotPOD));
  eventSetup.put(std::move(bs));
}

DEFINE_FWK_EVENTSETUP_MODULE(BeamSpotESProducer);
