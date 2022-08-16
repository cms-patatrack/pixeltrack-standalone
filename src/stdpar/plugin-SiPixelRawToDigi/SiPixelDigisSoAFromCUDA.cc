#include "CUDADataFormats/SiPixelDigis.h"
#include "DataFormats/SiPixelDigisSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/EDProducer.h"
#include "Framework/PluginFactory.h"

class SiPixelDigisSoAFromCUDA : public edm::EDProducer {
public:
  explicit SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg);
  ~SiPixelDigisSoAFromCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<SiPixelDigis> digiGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;
};

SiPixelDigisSoAFromCUDA::SiPixelDigisSoAFromCUDA(edm::ProductRegistry& reg)
    : digiGetToken_(reg.consumes<SiPixelDigis>()), digiPutToken_(reg.produces<SiPixelDigisSoA>()) {}

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
  const auto& gpuDigis = iEvent.get(digiGetToken_);
  iEvent.emplace(digiPutToken_, gpuDigis.nDigis(), gpuDigis.pdigi(), gpuDigis.rawIdArr(), gpuDigis.adc(), gpuDigis.clus());
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromCUDA);
