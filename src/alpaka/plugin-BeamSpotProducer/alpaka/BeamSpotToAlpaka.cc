#include <utility>

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/alpaka/BeamSpotAlpaka.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotToAlpaka : public edm::EDProducer {
  public:
    explicit BeamSpotToAlpaka(edm::ProductRegistry& reg);
    ~BeamSpotToAlpaka() override = default;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    const edm::EDPutTokenT<cms::alpakatools::Product<Queue, BeamSpotAlpaka>> bsPutToken_;

    cms::alpakatools::host_buffer<BeamSpotPOD> bsHost_;
  };

  BeamSpotToAlpaka::BeamSpotToAlpaka(edm::ProductRegistry& reg)
      : bsPutToken_{reg.produces<cms::alpakatools::Product<Queue, BeamSpotAlpaka>>()},
        bsHost_{cms::alpakatools::make_host_buffer<BeamSpotPOD, Platform>()} {}

  void BeamSpotToAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    *bsHost_ = iSetup.get<BeamSpotPOD>();

    cms::alpakatools::ScopedContextProduce<Queue> ctx{iEvent.streamID()};

    BeamSpotAlpaka bsDevice(ctx.stream());
    alpaka::memcpy(ctx.stream(), bsDevice.buf(), bsHost_);

    ctx.emplace(iEvent, bsPutToken_, std::move(bsDevice));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(BeamSpotToAlpaka);
