#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelROCsStatusAndMappingWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelROCsStatusAndMappingWrapper_h

#include "CUDACore/ESProduct.h"
#include "CUDACore/HostAllocator.h"
#include "CUDACore/device_unique_ptr.h"
#include "CondFormats/SiPixelROCsStatusAndMapping.h"

#include <cuda_runtime.h>

#include <set>

class SiPixelROCsStatusAndMappingWrapper {
public:
  explicit SiPixelROCsStatusAndMappingWrapper(SiPixelROCsStatusAndMapping const &cablingMap,
                                          std::vector<unsigned char> modToUnp);
  ~SiPixelROCsStatusAndMappingWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelROCsStatusAndMapping *getGPUProductAsync(cudaStream_t cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(cudaStream_t cudaStream) const;

private:
  std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> modToUnpDefault;
  bool hasQuality_;

  SiPixelROCsStatusAndMapping *cablingMapHost = nullptr;  // pointer to struct in CPU

  struct GPUData {
    ~GPUData();
    SiPixelROCsStatusAndMapping *cablingMapDevice = nullptr;  // pointer to struct in GPU
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    ~ModulesToUnpack();
    unsigned char *modToUnpDefault = nullptr;  // pointer to GPU
  };
  cms::cuda::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif
