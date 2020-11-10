#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include <cuda_runtime.h>

#include "CUDACore/SimpleVector.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/managed_unique_ptr.h"
#include "DataFormats/PixelErrors.h"

class SiPixelDigiErrorsCUDA {
public:
  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cudaStream_t stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::cuda::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
  cms::cuda::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
  cms::cuda::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
  using HostDataError =
      std::pair<cms::cuda::SimpleVector<PixelErrorCompact>, cms::cuda::host::unique_ptr<PixelErrorCompact[]>>;
  HostDataError dataErrorToHostAsync(cudaStream_t stream) const;

  void copyErrorToHostAsync(cudaStream_t stream);
#else
  void prefetchAsync(int device, cudaStream_t stream) const;
#endif

private:
#ifdef CUDAUVM_DISABLE_MANAGED_CLUSTERING
  cms::cuda::device::unique_ptr<PixelErrorCompact[]> data_d;
  cms::cuda::device::unique_ptr<cms::cuda::SimpleVector<PixelErrorCompact>> error_d;
  cms::cuda::host::unique_ptr<cms::cuda::SimpleVector<PixelErrorCompact>> error_h;
#else
  cms::cuda::managed::unique_ptr<PixelErrorCompact[]> data_d;
  cms::cuda::managed::unique_ptr<cms::cuda::SimpleVector<PixelErrorCompact>> error_d;
  size_t maxFedWords_;
#endif

  PixelFormatterErrors formatterErrors_h;
};

#endif
