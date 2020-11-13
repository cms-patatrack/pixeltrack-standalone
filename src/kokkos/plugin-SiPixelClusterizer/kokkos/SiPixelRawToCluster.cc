#include "KokkosDataFormats/SiPixelClustersKokkos.h"
#include "KokkosDataFormats/SiPixelDigisKokkos.h"
#include "KokkosDataFormats/SiPixelDigiErrorsKokkos.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"
#include "CondFormats/SiPixelFedIds.h"
#include "DataFormats/PixelErrors.h"
#include "DataFormats/FEDNumbering.h"
#include "DataFormats/FEDRawData.h"
#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "../ErrorChecker.h"
#include "SiPixelRawToClusterGPUKernel.h"

#include "KokkosCore/kokkosConfig.h"

#include <memory>
#include <string>
#include <vector>

namespace KOKKOS_NAMESPACE {
  class SiPixelRawToCluster : public edm::EDProducer {
  public:
    explicit SiPixelRawToCluster(edm::ProductRegistry& reg);
    ~SiPixelRawToCluster() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
    edm::EDPutTokenT<SiPixelDigisKokkos<KokkosExecSpace>> digiPutToken_;
    edm::EDPutTokenT<SiPixelDigiErrorsKokkos<KokkosExecSpace>> digiErrorPutToken_;
    edm::EDPutTokenT<SiPixelClustersKokkos<KokkosExecSpace>> clusterPutToken_;

    pixelgpudetails::SiPixelRawToClusterGPUKernel gpuAlgo_;
    std::unique_ptr<pixelgpudetails::SiPixelRawToClusterGPUKernel::WordFedAppender> wordFedAppender_;
    PixelFormatterErrors errors_;

    const bool isRun2_;
    const bool includeErrors_;
    const bool useQuality_;
  };

  SiPixelRawToCluster::SiPixelRawToCluster(edm::ProductRegistry& reg)
      : rawGetToken_(reg.consumes<FEDRawDataCollection>()),
        digiPutToken_(reg.produces<SiPixelDigisKokkos<KokkosExecSpace>>()),
        clusterPutToken_(reg.produces<SiPixelClustersKokkos<KokkosExecSpace>>()),
        isRun2_(true),
        includeErrors_(true),
        useQuality_(true) {
    if (includeErrors_) {
      digiErrorPutToken_ = reg.produces<SiPixelDigiErrorsKokkos<KokkosExecSpace>>();
    }

    wordFedAppender_ = std::make_unique<pixelgpudetails::SiPixelRawToClusterGPUKernel::WordFedAppender>();
  }

  void SiPixelRawToCluster::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto const& hgpuMap = iSetup.get<SiPixelFedCablingMapGPUWrapper<KokkosExecSpace>>();
    if (hgpuMap.hasQuality() != useQuality_) {
      throw std::runtime_error("UseQuality of the module (" + std::to_string(useQuality_) +
                               ") differs the one from SiPixelFedCablingMapGPUWrapper. Please fix your configuration.");
    }
    // get the GPU product already here so that the async transfer can begin
    const auto& gpuMap = hgpuMap.cablingMap();
    const auto& gpuModulesToUnpack = iSetup.get<Kokkos::View<const unsigned char*, KokkosExecSpace>>();
    auto const& gpuGains = iSetup.get<SiPixelGainForHLTonGPU<KokkosExecSpace>>();

    auto const& fedIds_ = iSetup.get<SiPixelFedIds>().fedIds();

    const auto& buffers = iEvent.get(rawGetToken_);

    errors_.clear();

    // GPU specific: Data extraction for RawToDigi GPU
    unsigned int wordCounterGPU = 0;
    unsigned int fedCounter = 0;
    bool errorsInEvent = false;

    // In CPU algorithm this loop is part of PixelDataFormatter::interpretRawData()
    ErrorChecker errorcheck;
    for (int fedId : fedIds_) {
      if (fedId == 40)
        continue;  // skip pilot blade data

      // for GPU
      // first 150 index stores the fedId and next 150 will store the
      // start index of word in that fed
      assert(fedId >= 1200);
      fedCounter++;

      // get event data for this fed
      const FEDRawData& rawData = buffers.FEDData(fedId);

      // GPU specific
      int nWords = rawData.size() / sizeof(uint64_t);
      if (nWords == 0) {
        continue;
      }

      // check CRC bit
      const uint64_t* trailer = reinterpret_cast<const uint64_t*>(rawData.data()) + (nWords - 1);
      if (not errorcheck.checkCRC(errorsInEvent, fedId, trailer, errors_)) {
        continue;
      }

      // check headers
      const uint64_t* header = reinterpret_cast<const uint64_t*>(rawData.data());
      header--;
      bool moreHeaders = true;
      while (moreHeaders) {
        header++;
        bool headerStatus = errorcheck.checkHeader(errorsInEvent, fedId, header, errors_);
        moreHeaders = headerStatus;
      }

      // check trailers
      bool moreTrailers = true;
      trailer++;
      while (moreTrailers) {
        trailer--;
        bool trailerStatus = errorcheck.checkTrailer(errorsInEvent, fedId, nWords, trailer, errors_);
        moreTrailers = trailerStatus;
      }

      const uint32_t* bw = (const uint32_t*)(header + 1);
      const uint32_t* ew = (const uint32_t*)(trailer);

      assert(0 == (ew - bw) % 2);
      wordFedAppender_->initializeWordFed(fedId, wordCounterGPU, bw, (ew - bw));
      wordCounterGPU += (ew - bw);

    }  // end of for loop
    gpuAlgo_.makeClustersAsync(isRun2_,
                               gpuMap,
                               gpuModulesToUnpack,
                               gpuGains,
                               *wordFedAppender_,
                               std::move(errors_),
                               wordCounterGPU,
                               fedCounter,
                               useQuality_,
                               includeErrors_,
                               false,  // debug
                               KokkosExecSpace());

    // TODO: synchronize explicitly for now
    KokkosExecSpace().fence();

    auto tmp = gpuAlgo_.getResults();
    iEvent.emplace(digiPutToken_, std::move(tmp.first));
    iEvent.emplace(clusterPutToken_, std::move(tmp.second));
    if (includeErrors_) {
      iEvent.emplace(digiErrorPutToken_, gpuAlgo_.getErrors());
    }
  }
}  // namespace KOKKOS_NAMESPACE

// define as framework plugin
DEFINE_FWK_KOKKOS_MODULE(SiPixelRawToCluster);
