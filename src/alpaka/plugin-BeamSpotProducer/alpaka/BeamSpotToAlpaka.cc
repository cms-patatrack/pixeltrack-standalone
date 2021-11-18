#include <utility>

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaDataFormats/BeamSpotAlpaka.h"
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
    const edm::EDPutTokenT<::cms::alpakatools::Product<Queue, BeamSpotAlpaka>> bsPutToken_;

    AlpakaHostBuf<BeamSpotPOD> bsHost_;
  };

  BeamSpotToAlpaka::BeamSpotToAlpaka(edm::ProductRegistry& reg)
      : bsPutToken_{reg.produces<::cms::alpakatools::Product<Queue, BeamSpotAlpaka>>()},
        bsHost_{::cms::alpakatools::allocHostBuf<BeamSpotPOD>(1u)} {
    alpaka::prepareForAsyncCopy(bsHost_);
  }

  void BeamSpotToAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    *alpaka::getPtrNative(bsHost_) = iSetup.get<BeamSpotPOD>();

    ::cms::alpakatools::ScopedContextProduce<Queue> ctx{iEvent.streamID()};

    BeamSpotAlpaka bsDevice(ctx.stream());
    alpaka::memcpy(ctx.stream(), bsDevice.buf(), bsHost_, 1u);

    ctx.emplace(iEvent, bsPutToken_, std::move(bsDevice));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(BeamSpotToAlpaka);
