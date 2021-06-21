#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include <cuda_runtime.h>

#include "CUDACore/SimpleVector.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "DataFormats/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelFormatterErrors.h"

class SiPixelDigiErrorsCUDA {
public:
  using SiPixelErrorCompactVector = cms::cuda::SimpleVector<SiPixelErrorCompact>;

  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, SiPixelFormatterErrors errors, cudaStream_t stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const SiPixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  SiPixelErrorCompactVector* error() { return error_d.get(); }
  SiPixelErrorCompactVector const* error() const { return error_d.get(); }

  using HostDataError = std::pair<SiPixelErrorCompactVector, cms::cuda::host::unique_ptr<SiPixelErrorCompact[]>>;
  HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

  void copyErrorToHostAsync(cudaStream_t stream);

private:
  cms::cuda::device::unique_ptr<SiPixelErrorCompact[]> data_d;
  cms::cuda::device::unique_ptr<SiPixelErrorCompactVector> error_d;
  cms::cuda::host::unique_ptr<SiPixelErrorCompactVector> error_h;
  SiPixelFormatterErrors formatterErrors_h;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
