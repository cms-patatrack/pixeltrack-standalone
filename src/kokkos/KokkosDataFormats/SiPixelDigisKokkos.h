#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/memoryTraits.h"
#include "KokkosCore/ViewHelpers.h"

template <typename MemorySpace>
class SiPixelDigisKokkos {
public:
  SiPixelDigisKokkos() = default;
  explicit SiPixelDigisKokkos(size_t maxFedWords)
      : xx_d{Kokkos::ViewAllocateWithoutInitializing("xx_d"), maxFedWords},
        yy_d{Kokkos::ViewAllocateWithoutInitializing("yy_d"), maxFedWords},
        adc_d{Kokkos::ViewAllocateWithoutInitializing("adc_d"), maxFedWords},
        moduleInd_d{Kokkos::ViewAllocateWithoutInitializing("moduleInd_d"), maxFedWords},
        clus_d{Kokkos::ViewAllocateWithoutInitializing("clus_d"), maxFedWords},
        pdigi_d{Kokkos::ViewAllocateWithoutInitializing("pdigi_d"), maxFedWords},
        rawIdArr_d{Kokkos::ViewAllocateWithoutInitializing("rawIdArr_d"), maxFedWords} {}
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

  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t*, MemorySpace>& xx() { return xx_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t*, MemorySpace>& yy() { return yy_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t*, MemorySpace>& adc() { return adc_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t*, MemorySpace>& moduleInd() { return moduleInd_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<int32_t*, MemorySpace>& clus() { return clus_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint32_t*, MemorySpace>& pdigi() { return pdigi_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint32_t*, MemorySpace>& rawIdArr() { return rawIdArr_d; }

  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t const*, MemorySpace> xx() const { return xx_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t const*, MemorySpace> yy() const { return yy_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t const*, MemorySpace> adc() const { return adc_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t const*, MemorySpace> moduleInd() const { return moduleInd_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<int32_t const*, MemorySpace> clus() const { return clus_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint32_t const*, MemorySpace> pdigi() const { return pdigi_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint32_t const*, MemorySpace> rawIdArr() const { return rawIdArr_d; }

  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t const*, MemorySpace> c_xx() const { return xx_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t const*, MemorySpace> c_yy() const { return yy_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t const*, MemorySpace> c_adc() const { return adc_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint16_t const*, MemorySpace> c_moduleInd() const { return moduleInd_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<int32_t const*, MemorySpace> c_clus() const { return clus_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint32_t const*, MemorySpace> c_pdigi() const { return pdigi_d; }
  KOKKOS_INLINE_FUNCTION Kokkos::View<uint32_t const*, MemorySpace> c_rawIdArr() const { return rawIdArr_d; }

  template <typename ExecSpace>
  auto adcToHostAsync(ExecSpace const& execSpace) const {
    auto host = cms::kokkos::create_mirror_view(adc_d);
    Kokkos::deep_copy(execSpace, host, adc_d);
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
    Kokkos::View<uint16_t const*, MemorySpace, Restrict> xx_;
    Kokkos::View<uint16_t const*, MemorySpace, Restrict> yy_;
    Kokkos::View<uint16_t const*, MemorySpace, Restrict> adc_;
    Kokkos::View<uint16_t const*, MemorySpace, Restrict> moduleInd_;
    Kokkos::View<int32_t const*, MemorySpace, Restrict> clus_;
  };

  DeviceConstView view() const { return DeviceConstView{xx_d, yy_d, adc_d, moduleInd_d, clus_d}; }

private:
  // These are consumed by downstream device code
  Kokkos::View<uint16_t*, MemorySpace> xx_d;         // local coordinates of each pixel
  Kokkos::View<uint16_t*, MemorySpace> yy_d;         //
  Kokkos::View<uint16_t*, MemorySpace> adc_d;        // ADC of each pixel
  Kokkos::View<uint16_t*, MemorySpace> moduleInd_d;  // module id of each pixel
  Kokkos::View<int32_t*, MemorySpace> clus_d;        // cluster id of each pixel
#ifdef TODO
  cms::cuda::device::unique_ptr<DeviceConstView> view_d;  // "me" pointer
#endif

  // These are for CPU output; should we (eventually) place them to a
  // separate product?
  Kokkos::View<uint32_t*, MemorySpace> pdigi_d;
  Kokkos::View<uint32_t*, MemorySpace> rawIdArr_d;

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif
