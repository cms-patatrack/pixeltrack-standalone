#include <fstream>

#include <sycl/sycl.hpp>

#include "SYCLCore/Product.h"
#include "SYCLCore/ScopedContext.h"
#include "SYCLCore/host_unique_ptr.h"
#include "SYCLDataFormats/BeamSpotSYCL.h"
#include "DataFormats/BeamSpotPOD.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

class BeamSpotToSYCL : public edm::EDProducer {
public:
  explicit BeamSpotToSYCL(edm::ProductRegistry& reg);
  ~BeamSpotToSYCL() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  const edm::EDPutTokenT<cms::sycltools::Product<BeamSpotSYCL>> bsPutToken_;
};

BeamSpotToSYCL::BeamSpotToSYCL(edm::ProductRegistry& reg)
    : bsPutToken_{reg.produces<cms::sycltools::Product<BeamSpotSYCL>>()} {}

void BeamSpotToSYCL::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::sycltools::ScopedContextProduce ctx{iEvent.streamID()};

  sycl::queue stream = ctx.stream();
  BeamSpotSYCL bsDevice(stream);

  // in CUDA this is done in the constructor, but we need the queue so we do it here
  cms::sycltools::host::unique_ptr<BeamSpotPOD> bsHost;
  bsHost = cms::sycltools::make_host_unique<BeamSpotPOD>(stream);
  *bsHost = iSetup.get<BeamSpotPOD>();

  stream.memcpy(bsDevice.ptr().get(), bsHost.get(), sizeof(BeamSpotPOD)).wait();
  ctx.emplace(iEvent, bsPutToken_, std::move(bsDevice));
}

DEFINE_FWK_MODULE(BeamSpotToSYCL);
