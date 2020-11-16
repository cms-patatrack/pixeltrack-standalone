#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include "DataFormats/PixelErrors.h"
#include "KokkosCore/SimpleVector.h"

#include "KokkosCore/kokkosConfig.h"

template <typename MemorySpace>
class SiPixelDigiErrorsKokkos {
public:
  SiPixelDigiErrorsKokkos() = default;
  template <typename ExecSpace>
  explicit SiPixelDigiErrorsKokkos(size_t maxFedWords, PixelFormatterErrors errors, ExecSpace const& execSpace)
      : data_d{Kokkos::ViewAllocateWithoutInitializing("data_d"), maxFedWords},
        error_d{Kokkos::ViewAllocateWithoutInitializing("error_d")},
        error_h{Kokkos::ViewAllocateWithoutInitializing("error_h")},
        formatterErrors_h{std::move(errors)} {
    error_h().construct(maxFedWords, data_d.data());
    assert(error_h().empty());
    assert(error_h().capacity() == static_cast<int>(maxFedWords));
    Kokkos::deep_copy(execSpace, error_d, error_h);
  }
  ~SiPixelDigiErrorsKokkos() = default;

  SiPixelDigiErrorsKokkos(const SiPixelDigiErrorsKokkos&) = delete;
  SiPixelDigiErrorsKokkos& operator=(const SiPixelDigiErrorsKokkos&) = delete;
  SiPixelDigiErrorsKokkos(SiPixelDigiErrorsKokkos&&) = default;
  SiPixelDigiErrorsKokkos& operator=(SiPixelDigiErrorsKokkos&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::kokkos::SimpleVector<PixelErrorCompact>* error() { return error_d.data(); }
  cms::kokkos::SimpleVector<PixelErrorCompact> const* error() const { return error_d.data(); }
  cms::kokkos::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.data(); }

#ifdef TODO
  using HostDataError = std::pair<cms::kokkos::SimpleVector<PixelErrorCompact>,
                                  typename Kokkos::View<PixelErrorCompact[], MemorySpace>::HostMirror>;
  HostDataError dataErrorToHostAsync() const {}

  void copyErrorToHostAsync() {}
#endif

private:
  Kokkos::View<PixelErrorCompact*, MemorySpace> data_d;
  Kokkos::View<cms::kokkos::SimpleVector<PixelErrorCompact>, MemorySpace> error_d;
  typename Kokkos::View<cms::kokkos::SimpleVector<PixelErrorCompact>, MemorySpace>::HostMirror error_h;
  PixelFormatterErrors formatterErrors_h;
};

#endif
