#include "SYCLCore/Product.h"
#include "SYCLDataFormats/SiPixelDigisSYCL.h"
#include "DataFormats/SiPixelDigisSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/EDProducer.h"
#include "Framework/PluginFactory.h"
#include "SYCLCore/ScopedContext.h"
#include "SYCLCore/host_unique_ptr.h"

class SiPixelDigisSoAFromSYCL : public edm::EDProducerExternalWork {
public:
  explicit SiPixelDigisSoAFromSYCL(edm::ProductRegistry& reg);
  ~SiPixelDigisSoAFromSYCL() override = default;

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<cms::sycltools::Product<SiPixelDigisSYCL>> digiGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  cms::sycltools::host::unique_ptr<uint32_t[]> pdigi_;
  cms::sycltools::host::unique_ptr<uint32_t[]> rawIdArr_;
  cms::sycltools::host::unique_ptr<uint16_t[]> adc_;
  cms::sycltools::host::unique_ptr<int32_t[]> clus_;

  size_t nDigis_;
};

SiPixelDigisSoAFromSYCL::SiPixelDigisSoAFromSYCL(edm::ProductRegistry& reg)
    : digiGetToken_(reg.consumes<cms::sycltools::Product<SiPixelDigisSYCL>>()),
      digiPutToken_(reg.produces<SiPixelDigisSoA>()) {}

void SiPixelDigisSoAFromSYCL::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Do the transfer in a SYCL stream parallel to the computation SYCL stream
  cms::sycltools::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};

  const auto& gpuDigis = ctx.get(iEvent, digiGetToken_);

  auto stream = ctx.stream();
  nDigis_ = gpuDigis.nDigis();
  pdigi_ = gpuDigis.pdigiToHostAsync(stream);
  rawIdArr_ = gpuDigis.rawIdArrToHostAsync(stream);
  adc_ = gpuDigis.adcToHostAsync(stream);
  clus_ = gpuDigis.clusToHostAsync(stream);
}

void SiPixelDigisSoAFromSYCL::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
DEFINE_FWK_MODULE(SiPixelDigisSoAFromSYCL);
