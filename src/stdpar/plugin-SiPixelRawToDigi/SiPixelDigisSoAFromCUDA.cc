#include "CUDACore/Product.h"
#include "CUDADataFormats/SiPixelDigis.h"
#include "DataFormats/SiPixelDigisSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/EDProducer.h"
#include "Framework/PluginFactory.h"
#include "CUDACore/ScopedContext.h"

class SiPixelDigisSoAFromCUDA : public edm::EDProducerExternalWork {
public:
  explicit SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg);
  ~SiPixelDigisSoAFromCUDA() override = default;

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigis>> digiGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  //See comments in constructor body
  uint32_t const* pdigi_{nullptr};
  uint32_t const* rawIdArr_{nullptr};
  uint16_t const* adc_{nullptr};
  int32_t const* clus_{nullptr};

  size_t nDigis_;
};

SiPixelDigisSoAFromCUDA::SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg)
    : digiGetToken_(reg.consumes<cms::cuda::Product<SiPixelDigis>>()), digiPutToken_(reg.produces<SiPixelDigisSoA>()) {}

void SiPixelDigisSoAFromCUDA::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Do the transfer in a CUDA stream parallel to the computation CUDA stream
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};

  const auto& gpuDigis = ctx.get(iEvent, digiGetToken_);

  nDigis_ = gpuDigis.nDigis();
  // This memory is already owned by a unique pointer, therefore class members cannot be unique_ptr themselves
  // *this must have a shorter lifetime than gpuDigis object retrieved from the context or we risk using dangling pointers
  pdigi_ = gpuDigis.pdigi();
  rawIdArr_ = gpuDigis.rawIdArr();
  adc_ = gpuDigis.adc();
  clus_ = gpuDigis.clus();
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
  iEvent.emplace(digiPutToken_, nDigis_, pdigi_, rawIdArr_, adc_, clus_);
  pdigi_ = nullptr;
  rawIdArr_ = nullptr;
  adc_ = nullptr;
  clus_ = nullptr;
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromCUDA);
