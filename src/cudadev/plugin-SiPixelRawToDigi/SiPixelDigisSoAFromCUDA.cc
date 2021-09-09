#include "CUDACore/EDProducer.h"
#include "CUDACore/Product.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "DataFormats/SiPixelDigisSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "CUDACore/host_unique_ptr.h"

class SiPixelDigisSoAFromCUDA : public cms::cuda::SynchronizingEDProducer {
public:
  explicit SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg);
  ~SiPixelDigisSoAFromCUDA() override = default;

private:
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, cms::cuda::AcquireContext& ctx) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup, cms::cuda::ProduceContext&) override;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  cms::cuda::host::unique_ptr<uint32_t[]> pdigi_;
  cms::cuda::host::unique_ptr<uint32_t[]> rawIdArr_;
  cms::cuda::host::unique_ptr<uint16_t[]> adc_;
  cms::cuda::host::unique_ptr<int32_t[]> clus_;

  size_t nDigis_;
};

SiPixelDigisSoAFromCUDA::SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg)
    : digiGetToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
      digiPutToken_(reg.produces<SiPixelDigisSoA>()) {}

void SiPixelDigisSoAFromCUDA::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      cms::cuda::AcquireContext& ctx) {
  const auto& gpuDigis = ctx.get(iEvent, digiGetToken_);

  nDigis_ = gpuDigis.nDigis();
  pdigi_ = gpuDigis.pdigiToHostAsync(ctx.stream());
  rawIdArr_ = gpuDigis.rawIdArrToHostAsync(ctx.stream());
  adc_ = gpuDigis.adcToHostAsync(ctx.stream());
  clus_ = gpuDigis.clusToHostAsync(ctx.stream());
}

void SiPixelDigisSoAFromCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup, cms::cuda::ProduceContext&) {
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
  iEvent.emplace(digiPutToken_, nDigis_, pdigi_.get(), rawIdArr_.get(), adc_.get(), clus_.get());

  pdigi_.reset();
  rawIdArr_.reset();
  adc_.reset();
  clus_.reset();
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromCUDA);
