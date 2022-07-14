#include <fstream>

#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDACore/copyAsync.h"
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
#include "CUDACore/host_noncached_unique_ptr.h"
#endif
#include "CUDADataFormats/BeamSpotCUDA.h"
#include "DataFormats/BeamSpotPOD.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

class BeamSpotToCUDA : public edm::EDProducer {
public:
  explicit BeamSpotToCUDA(edm::ProductRegistry& reg);
  ~BeamSpotToCUDA() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDPutTokenT<cms::cuda::Product<BeamSpotCUDA>> bsPutToken_;

#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  cms::cuda::host::noncached::unique_ptr<BeamSpotPOD> bsHost;
#endif
};

BeamSpotToCUDA::BeamSpotToCUDA(edm::ProductRegistry& reg)
    : bsPutToken_(reg.produces<cms::cuda::Product<BeamSpotCUDA>>())
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
      ,
      bsHost(cms::cuda::make_host_noncached_unique<BeamSpotPOD>(cudaHostAllocWriteCombined))
#endif
{
}

void BeamSpotToCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  *bsHost = iSetup.get<BeamSpotPOD>();
#else
  auto const& bs = iSetup.get<BeamSpotPOD>();
#endif

  cms::cuda::ScopedContextProduce ctx{iEvent.streamID()};

  BeamSpotCUDA bsDevice(ctx.stream());
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  cms::cuda::copyAsync(bsDevice.ptr(), bsHost, ctx.stream());
#else
  *(bsDevice.data()) = bs;
  bsDevice.memAdviseAndPrefetch(ctx.device(), ctx.stream());
#endif  // CUDAUVM_DISABLE_MANAGED_BEAMSPOT

  ctx.emplace(iEvent, bsPutToken_, std::move(bsDevice));
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);
