#ifndef SYCLDataFormats_SiPixelDigi_interface_SiPixelDigiErrorsSYCL_h
#define SYCLDataFormats_SiPixelDigi_interface_SiPixelDigiErrorsSYCL_h

#include <CL/sycl.hpp>

#include "SYCLCore/SimpleVector.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "DataFormats/PixelErrors.h"

class SiPixelDigiErrorsSYCL {
public:
  SiPixelDigiErrorsSYCL() = default;
  explicit SiPixelDigiErrorsSYCL(size_t maxFedWords, PixelFormatterErrors errors, sycl::queue stream);
  ~SiPixelDigiErrorsSYCL() = default;

  SiPixelDigiErrorsSYCL(const SiPixelDigiErrorsSYCL&) = delete;
  SiPixelDigiErrorsSYCL& operator=(const SiPixelDigiErrorsSYCL&) = delete;
  SiPixelDigiErrorsSYCL(SiPixelDigiErrorsSYCL&&) = default;
  SiPixelDigiErrorsSYCL& operator=(SiPixelDigiErrorsSYCL&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::sycltools::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
  cms::sycltools::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
  cms::sycltools::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

  using HostDataError =
      std::pair<cms::sycltools::SimpleVector<PixelErrorCompact>, cms::sycltools::host::unique_ptr<PixelErrorCompact[]>>;
  HostDataError dataErrorToHostAsync(sycl::queue stream) const;

  void copyErrorToHostAsync(sycl::queue stream);

private:
  cms::sycltools::device::unique_ptr<PixelErrorCompact[]> data_d;
  cms::sycltools::device::unique_ptr<cms::sycltools::SimpleVector<PixelErrorCompact>> error_d;
  cms::sycltools::host::unique_ptr<cms::sycltools::SimpleVector<PixelErrorCompact>> error_h;
  PixelFormatterErrors formatterErrors_h;
};

#endif
