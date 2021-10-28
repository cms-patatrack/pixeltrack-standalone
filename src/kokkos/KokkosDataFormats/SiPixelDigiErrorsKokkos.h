#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include "DataFormats/PixelErrors.h"
#include "KokkosCore/SimpleVector.h"

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/deep_copy.h"
#include "KokkosCore/shared_ptr.h"

template <typename MemorySpace>
class SiPixelDigiErrorsKokkos {
public:
  SiPixelDigiErrorsKokkos() = default;
  template <typename ExecSpace>
  explicit SiPixelDigiErrorsKokkos(size_t maxFedWords, PixelFormatterErrors errors, ExecSpace const& execSpace)
      : data_d{cms::kokkos::make_shared<PixelErrorCompact[], MemorySpace>(maxFedWords)},
        error_d{cms::kokkos::make_shared<cms::kokkos::SimpleVector<PixelErrorCompact>, MemorySpace>()},
        error_h{cms::kokkos::make_mirror_shared(error_d)},
        formatterErrors_h{std::move(errors)} {
    error_h->construct(maxFedWords, data_d.get());
    assert(error_h->empty());
    assert(error_h->capacity() == static_cast<int>(maxFedWords));
    cms::kokkos::deep_copy(execSpace, error_d, error_h);
  }
  ~SiPixelDigiErrorsKokkos() = default;

  SiPixelDigiErrorsKokkos(const SiPixelDigiErrorsKokkos&) = delete;
  SiPixelDigiErrorsKokkos& operator=(const SiPixelDigiErrorsKokkos&) = delete;
  SiPixelDigiErrorsKokkos(SiPixelDigiErrorsKokkos&&) = default;
  SiPixelDigiErrorsKokkos& operator=(SiPixelDigiErrorsKokkos&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::kokkos::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
  cms::kokkos::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
  cms::kokkos::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

#ifdef TODO
  using HostDataError = std::pair<cms::kokkos::SimpleVector<PixelErrorCompact>,
                                  typename Kokkos::View<PixelErrorCompact[], MemorySpace>::HostMirror>;
  HostDataError dataErrorToHostAsync() const {}

  void copyErrorToHostAsync() {}
#endif

private:
  cms::kokkos::shared_ptr<PixelErrorCompact[], MemorySpace> data_d;
  cms::kokkos::shared_ptr<cms::kokkos::SimpleVector<PixelErrorCompact>, MemorySpace> error_d;
  cms::kokkos::shared_ptr<cms::kokkos::SimpleVector<PixelErrorCompact>, cms::kokkos::HostMirrorSpace_t<MemorySpace>>
      error_h;
  PixelFormatterErrors formatterErrors_h;
};

#endif
