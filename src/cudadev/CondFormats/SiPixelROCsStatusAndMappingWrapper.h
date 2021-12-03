#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelROCsStatusAndMappingWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelROCsStatusAndMappingWrapper_h

#include "CUDACore/ESProduct.h"
#include "CUDACore/HostAllocator.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/device_unique_ptr.h"
#include "CondFormats/SiPixelROCsStatusAndMapping.h"

#include <cuda_runtime.h>

#include <set>

class SiPixelROCsStatusAndMappingWrapper {
public:
  /* This is using a layout as the size is needed. TODO: use views when views start embedding size. */
  explicit SiPixelROCsStatusAndMappingWrapper(SiPixelROCsStatusAndMappingLayout const &cablingMap,
                                          std::vector<unsigned char> modToUnp);

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  SiPixelROCsStatusAndMappingConstView getGPUProductAsync(cudaStream_t cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(cudaStream_t cudaStream) const;

private:
  std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> modToUnpDefault;
  bool hasQuality_;

  cms::cuda::host::unique_ptr<std::byte[]> cablingMapHostBuffer;  // host pined memory for cabling map.

  struct GPUData {
    void allocate(size_t size, cudaStream_t stream) {
      cablingMapDeviceBuffer = cms::cuda::make_device_unique<std::byte[]>(
              SiPixelROCsStatusAndMappingLayout::computeDataSize(size), stream);
      cablingMapDevice = SiPixelROCsStatusAndMappingLayout(cablingMapDeviceBuffer.get(), size);
    }
    cms::cuda::device::unique_ptr<std::byte[]> cablingMapDeviceBuffer;
    SiPixelROCsStatusAndMappingLayout cablingMapDevice = SiPixelROCsStatusAndMappingLayout(nullptr, 0); // map struct in GPU
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    cms::cuda::device::unique_ptr<unsigned char []> modToUnpDefault;  // pointer to GPU
  };
  cms::cuda::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif
