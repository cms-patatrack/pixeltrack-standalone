#include "CUDACore/Product.h"
#include "CUDADataFormats/BeamSpotCUDA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CUDACore/ScopedContext.h"
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
#include "CUDACore/host_noncached_unique_ptr.h"
#endif

#include <cuda_runtime.h>

#include <fstream>

class BeamSpotToCUDA : public edm::EDProducer {
public:
  explicit BeamSpotToCUDA(edm::ProductRegistry& reg);
  ~BeamSpotToCUDA() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDPutTokenT<cms::cuda::Product<BeamSpotCUDA>> bsPutToken_;

#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  cms::cuda::host::noncached::unique_ptr<BeamSpotCUDA::Data> bsHost;
#endif
};

BeamSpotToCUDA::BeamSpotToCUDA(edm::ProductRegistry& reg)
    : bsPutToken_(reg.produces<cms::cuda::Product<BeamSpotCUDA>>())
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
      ,
      bsHost(cms::cuda::make_host_noncached_unique<BeamSpotCUDA::Data>(cudaHostAllocWriteCombined))
#endif
{
}

void BeamSpotToCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  *bsHost = iSetup.get<BeamSpotCUDA::Data>();
#else
  auto const& bs = iSetup.get<BeamSpotCUDA::Data>();
#endif

  cms::cuda::ScopedContextProduce ctx{iEvent.streamID()};

#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  ctx.emplace(iEvent, bsPutToken_, bsHost.get(), ctx.stream());
#else
  ctx.emplace(iEvent, bsPutToken_, bs, ctx.device(), ctx.stream());
#endif
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);
