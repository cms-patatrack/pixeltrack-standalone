#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelDigisAlpaka {
  public:
    SiPixelDigisAlpaka() = default;
    explicit SiPixelDigisAlpaka(size_t maxFedWords)
        : xx_d{cms::alpakatools::allocDeviceBuf<uint16_t>(maxFedWords)},
          yy_d{cms::alpakatools::allocDeviceBuf<uint16_t>(maxFedWords)},
          adc_d{cms::alpakatools::allocDeviceBuf<uint16_t>(maxFedWords)},
          moduleInd_d{cms::alpakatools::allocDeviceBuf<uint16_t>(maxFedWords)},
          clus_d{cms::alpakatools::allocDeviceBuf<int32_t>(maxFedWords)},
          pdigi_d{cms::alpakatools::allocDeviceBuf<uint32_t>(maxFedWords)},
          rawIdArr_d{cms::alpakatools::allocDeviceBuf<uint32_t>(maxFedWords)} {}
    ~SiPixelDigisAlpaka() = default;

    SiPixelDigisAlpaka(const SiPixelDigisAlpaka &) = delete;
    SiPixelDigisAlpaka &operator=(const SiPixelDigisAlpaka &) = delete;
    SiPixelDigisAlpaka(SiPixelDigisAlpaka &&) = default;
    SiPixelDigisAlpaka &operator=(SiPixelDigisAlpaka &&) = default;

    void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
      nModules_h = nModules;
      nDigis_h = nDigis;
    }

    uint32_t nModules() const { return nModules_h; }
    uint32_t nDigis() const { return nDigis_h; }

    uint16_t *xx() { return alpaka::getPtrNative(xx_d); }
    uint16_t *yy() { return alpaka::getPtrNative(yy_d); }
    uint16_t *adc() { return alpaka::getPtrNative(adc_d); }
    uint16_t *moduleInd() { return alpaka::getPtrNative(moduleInd_d); }
    int32_t *clus() { return alpaka::getPtrNative(clus_d); }
    uint32_t *pdigi() { return alpaka::getPtrNative(pdigi_d); }
    uint32_t *rawIdArr() { return alpaka::getPtrNative(rawIdArr_d); }

    uint16_t const *xx() const { return alpaka::getPtrNative(xx_d); }
    uint16_t const *yy() const { return alpaka::getPtrNative(yy_d); }
    uint16_t const *adc() const { return alpaka::getPtrNative(adc_d); }
    uint16_t const *moduleInd() const { return alpaka::getPtrNative(moduleInd_d); }
    int32_t const *clus() const { return alpaka::getPtrNative(clus_d); }
    uint32_t const *pdigi() const { return alpaka::getPtrNative(pdigi_d); }
    uint32_t const *rawIdArr() const { return alpaka::getPtrNative(rawIdArr_d); }

    uint16_t const *c_xx() const { return alpaka::getPtrNative(xx_d); }
    uint16_t const *c_yy() const { return alpaka::getPtrNative(yy_d); }
    uint16_t const *c_adc() const { return alpaka::getPtrNative(adc_d); }
    uint16_t const *c_moduleInd() const { return alpaka::getPtrNative(moduleInd_d); }
    int32_t const *c_clus() const { return alpaka::getPtrNative(clus_d); }
    uint32_t const *c_pdigi() const { return alpaka::getPtrNative(pdigi_d); }
    uint32_t const *c_rawIdArr() const { return alpaka::getPtrNative(rawIdArr_d); }

    // TO DO: nothing async in here for now... Pass the queue as argument instead, and don't wait anymore!
    auto adcToHostAsync(Queue &queue) const {
      auto ret = cms::alpakatools::allocHostBuf<uint16_t>(nDigis());
      alpaka::memcpy(queue, ret, adc_d, nDigis());
      return ret;
    }

#ifdef TODO
    cms::cuda::host::unique_ptr<int32_t[]> clusToHostAsync(cudaStream_t stream) const;
    cms::cuda::host::unique_ptr<uint32_t[]> pdigiToHostAsync(cudaStream_t stream) const;
    cms::cuda::host::unique_ptr<uint32_t[]> rawIdArrToHostAsync(cudaStream_t stream) const;
#endif

    class DeviceConstView {
    public:
      // TO DO: removed __ldg, check impact on perf with src/cuda.
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint16_t xx(int i) const { return xx_[i]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint16_t yy(int i) const { return yy_[i]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint16_t adc(int i) const { return adc_[i]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint16_t moduleInd(int i) const { return moduleInd_[i]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t clus(int i) const { return clus_[i]; }

      friend class SiPixelDigisAlpaka;

      // private:
      uint16_t const *__restrict__ xx_;
      uint16_t const *__restrict__ yy_;
      uint16_t const *__restrict__ adc_;
      uint16_t const *__restrict__ moduleInd_;
      int32_t const *__restrict__ clus_;
    };

    const DeviceConstView view() const { return DeviceConstView{c_xx(), c_yy(), c_adc(), c_moduleInd(), c_clus()}; }

  private:
    // These are consumed by downstream device code
    AlpakaDeviceBuf<uint16_t> xx_d;         // local coordinates of each pixel
    AlpakaDeviceBuf<uint16_t> yy_d;         //
    AlpakaDeviceBuf<uint16_t> adc_d;        // ADC of each pixel
    AlpakaDeviceBuf<uint16_t> moduleInd_d;  // module id of each pixel
    AlpakaDeviceBuf<int32_t> clus_d;        // cluster id of each pixel

    // These are for CPU output; should we (eventually) place them to a
    // separate product?
    AlpakaDeviceBuf<uint32_t> pdigi_d;
    AlpakaDeviceBuf<uint32_t> rawIdArr_d;

    uint32_t nModules_h = 0;
    uint32_t nDigis_h = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
