#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include <hip/hip_runtime.h>

#include "CUDACore/SimpleVector.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "DataFormats/PixelErrors.h"

class SiPixelDigiErrorsCUDA {
public:
  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, hipStream_t stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::hip::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
  cms::hip::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
  cms::hip::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

  using HostDataError =
      std::pair<cms::hip::SimpleVector<PixelErrorCompact>, cms::hip::host::unique_ptr<PixelErrorCompact[]>>;
  HostDataError dataErrorToHostAsync(hipStream_t stream) const;

  void copyErrorToHostAsync(hipStream_t stream);

private:
  cms::hip::device::unique_ptr<PixelErrorCompact[]> data_d;
  cms::hip::device::unique_ptr<cms::hip::SimpleVector<PixelErrorCompact>> error_d;
  cms::hip::host::unique_ptr<cms::hip::SimpleVector<PixelErrorCompact>> error_h;
  PixelFormatterErrors formatterErrors_h;
};

#endif
