#include "CUDACore/Product.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "DataFormats/SiPixelDigisSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/EDProducer.h"
#include "Framework/PluginFactory.h"
#include "CUDACore/ScopedContext.h"
#include "CUDACore/host_unique_ptr.h"

class SiPixelDigisSoAFromCUDA : public edm::EDProducerExternalWork {
public:
  explicit SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg);
  ~SiPixelDigisSoAFromCUDA() override = default;

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  SiPixelDigisCUDA::HostStoreAndBuffer digis_;

  size_t nDigis_;
};

SiPixelDigisSoAFromCUDA::SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg)
    : digiGetToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
      digiPutToken_(reg.produces<SiPixelDigisSoA>()) {}

void SiPixelDigisSoAFromCUDA::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Do the transfer in a CUDA stream parallel to the computation CUDA stream
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};

  const auto& gpuDigis = ctx.get(iEvent, digiGetToken_);

  nDigis_ = gpuDigis.nDigis();
  digis_ = gpuDigis.dataToHostAsync(ctx.stream());
}

void SiPixelDigisSoAFromCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // The following line copies the data from the pinned host memory to
  // regular host memory. In principle that feels unnecessary (why not
  // just use the pinned host memory?). There are a few arguments for
  // doing it though
  // - Now can release the pinned host memory back to the (caching) allocator
  //   * if we'd like to keep the pinned memory, we'd need to also
  //     keep the CUDA stream around as long as that, or allow pinned
  //     host memory to be allocated without a CUDA stream
  // - What if a CPU algorithm would produce the same SoA? We can't
  //   use cudaMallocHost without a GPU...
  auto dv = digis_.store();
  iEvent.emplace(digiPutToken_, nDigis_, dv.pdigi(), dv.rawIdArr(), dv.adc(), dv.clus());

  digis_.reset();
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromCUDA);
