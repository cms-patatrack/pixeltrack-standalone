#include "CUDACore/Product.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "DataFormats/SiPixelDigisSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/EDProducer.h"
#include "Framework/PluginFactory.h"
#include "CUDACore/ScopedContext.h"
#include "CUDACore/host_unique_ptr.h"

struct SiPixelDigisSoAFromCUDA_AsyncState {
  cms::cuda::host::unique_ptr<uint32_t[]> pdigi;
  cms::cuda::host::unique_ptr<uint32_t[]> rawIdArr;
  cms::cuda::host::unique_ptr<uint16_t[]> adc;
  cms::cuda::host::unique_ptr<int32_t[]> clus;
  size_t nDigis;
};

class SiPixelDigisSoAFromCUDA : public edm::EDProducerExternalWork<SiPixelDigisSoAFromCUDA_AsyncState> {
public:
  explicit SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg);
  ~SiPixelDigisSoAFromCUDA() override = default;

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder,
               AsyncState& state) const override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup, AsyncState& state) override;

  const edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiGetToken_;
  const edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;
};

SiPixelDigisSoAFromCUDA::SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg)
    : digiGetToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
      digiPutToken_(reg.produces<SiPixelDigisSoA>()) {}

void SiPixelDigisSoAFromCUDA::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                      AsyncState& state) const {
  // Do the transfer in a CUDA stream parallel to the computation CUDA stream
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};

  const auto& gpuDigis = ctx.get(iEvent, digiGetToken_);
  state.pdigi = gpuDigis.pdigiToHostAsync(ctx.stream());
  state.rawIdArr = gpuDigis.rawIdArrToHostAsync(ctx.stream());
  state.adc = gpuDigis.adcToHostAsync(ctx.stream());
  state.clus = gpuDigis.clusToHostAsync(ctx.stream());
  state.nDigis = gpuDigis.nDigis();
}

void SiPixelDigisSoAFromCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup, AsyncState& state) {
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
  iEvent.emplace(
      digiPutToken_, state.nDigis, state.pdigi.get(), state.rawIdArr.get(), state.adc.get(), state.clus.get());
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromCUDA);
