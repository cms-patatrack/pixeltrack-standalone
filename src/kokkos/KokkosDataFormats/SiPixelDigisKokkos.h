#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/memoryTraits.h"
#include "KokkosCore/ViewHelpers.h"
#include "KokkosCore/deep_copy.h"
#include "KokkosCore/shared_ptr.h"

template <typename MemorySpace>
class SiPixelDigisKokkos {
public:
  template <typename T>
  using View = Kokkos::View<T, MemorySpace, RestrictUnmanaged>;

  SiPixelDigisKokkos() = default;
  explicit SiPixelDigisKokkos(size_t maxFedWords)
      : xx_d{cms::kokkos::make_shared<uint16_t[], MemorySpace>(maxFedWords)},
        yy_d{cms::kokkos::make_shared<uint16_t[], MemorySpace>(maxFedWords)},
        adc_d{cms::kokkos::make_shared<uint16_t[], MemorySpace>(maxFedWords)},
        moduleInd_d{cms::kokkos::make_shared<uint16_t[], MemorySpace>(maxFedWords)},
        clus_d{cms::kokkos::make_shared<int32_t[], MemorySpace>(maxFedWords)},
        pdigi_d{cms::kokkos::make_shared<uint32_t[], MemorySpace>(maxFedWords)},
        rawIdArr_d{cms::kokkos::make_shared<uint32_t[], MemorySpace>(maxFedWords)} {}
  ~SiPixelDigisKokkos() = default;

  SiPixelDigisKokkos(const SiPixelDigisKokkos&) = delete;
  SiPixelDigisKokkos& operator=(const SiPixelDigisKokkos&) = delete;
  SiPixelDigisKokkos(SiPixelDigisKokkos&&) = default;
  SiPixelDigisKokkos& operator=(SiPixelDigisKokkos&&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  View<uint16_t*> xx() { return cms::kokkos::to_view(xx_d); }
  View<uint16_t*> yy() { return cms::kokkos::to_view(yy_d); }
  View<uint16_t*> adc() { return cms::kokkos::to_view(adc_d); }
  View<uint16_t*> moduleInd() { return cms::kokkos::to_view(moduleInd_d); }
  View<int32_t*> clus() { return cms::kokkos::to_view(clus_d); }
  View<uint32_t*> pdigi() { return cms::kokkos::to_view(pdigi_d); }
  View<uint32_t*> rawIdArr() { return cms::kokkos::to_view(rawIdArr_d); }

  View<uint16_t const*> xx() const { return cms::kokkos::to_view(xx_d); }
  View<uint16_t const*> yy() const { return cms::kokkos::to_view(yy_d); }
  View<uint16_t const*> adc() const { return cms::kokkos::to_view(adc_d); }
  View<uint16_t const*> moduleInd() const { return cms::kokkos::to_view(moduleInd_d); }
  View<int32_t const*> clus() const { return cms::kokkos::to_view(clus_d); }
  View<uint32_t const*> pdigi() const { return cms::kokkos::to_view(pdigi_d); }
  View<uint32_t const*> rawIdArr() const { return cms::kokkos::to_view(rawIdArr_d); }

  View<uint16_t const*> c_xx() const { return cms::kokkos::to_view(xx_d); }
  View<uint16_t const*> c_yy() const { return cms::kokkos::to_view(yy_d); }
  View<uint16_t const*> c_adc() const { return cms::kokkos::to_view(adc_d); }
  View<uint16_t const*> c_moduleInd() const { return cms::kokkos::to_view(moduleInd_d); }
  View<int32_t const*> c_clus() const { return cms::kokkos::to_view(clus_d); }
  View<uint32_t const*> c_pdigi() const { return cms::kokkos::to_view(pdigi_d); }
  View<uint32_t const*> c_rawIdArr() const { return cms::kokkos::to_view(rawIdArr_d); }

  template <typename ExecSpace>
  auto adcToHostAsync(ExecSpace const& execSpace) const {
    auto host = cms::kokkos::make_mirror_shared(adc_d);
    cms::kokkos::deep_copy(execSpace, host, adc_d);
    return host;
  }
#ifdef TODO
  cms::cuda::host::unique_ptr<int32_t[]> clusToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> pdigiToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> rawIdArrToHostAsync(cudaStream_t stream) const;
#endif

  class DeviceConstView {
  public:
    KOKKOS_FORCEINLINE_FUNCTION uint16_t xx(int i) const { return xx_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint16_t yy(int i) const { return yy_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint16_t adc(int i) const { return adc_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION uint16_t moduleInd(int i) const { return moduleInd_[i]; }
    KOKKOS_FORCEINLINE_FUNCTION int32_t clus(int i) const { return clus_[i]; }

    friend class SiPixelDigisKokkos;

    // private:
    Kokkos::View<uint16_t const*, MemorySpace, RestrictUnmanaged> xx_;
    Kokkos::View<uint16_t const*, MemorySpace, RestrictUnmanaged> yy_;
    Kokkos::View<uint16_t const*, MemorySpace, RestrictUnmanaged> adc_;
    Kokkos::View<uint16_t const*, MemorySpace, RestrictUnmanaged> moduleInd_;
    Kokkos::View<int32_t const*, MemorySpace, RestrictUnmanaged> clus_;
  };

  DeviceConstView view() const { return DeviceConstView{xx(), yy(), adc(), moduleInd(), clus()}; }

private:
  // These are consumed by downstream device code
  cms::kokkos::shared_ptr<uint16_t[], MemorySpace> xx_d;         // local coordinates of each pixel
  cms::kokkos::shared_ptr<uint16_t[], MemorySpace> yy_d;         //
  cms::kokkos::shared_ptr<uint16_t[], MemorySpace> adc_d;        // ADC of each pixel
  cms::kokkos::shared_ptr<uint16_t[], MemorySpace> moduleInd_d;  // module id of each pixel
  cms::kokkos::shared_ptr<int32_t[], MemorySpace> clus_d;        // cluster id of each pixel
#ifdef TODO
  cms::cuda::device::unique_ptr<DeviceConstView> view_d;  // "me" pointer
#endif

  // These are for CPU output; should we (eventually) place them to a
  // separate product?
  cms::kokkos::shared_ptr<uint32_t[], MemorySpace> pdigi_d;
  cms::kokkos::shared_ptr<uint32_t[], MemorySpace> rawIdArr_d;

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif
