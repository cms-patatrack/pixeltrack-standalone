#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CUDACore/ESProduct.h"
#include "CUDACore/HostAllocator.h"
#include "CUDACore/device_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include <cuda_runtime.h>

#include <set>

class SiPixelFedCablingMapGPUWrapper {
public:
  explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const &cablingMap,
                                          std::vector<unsigned char> modToUnp);
  ~SiPixelFedCablingMapGPUWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU *getGPUProductAsync(cudaStream_t cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(cudaStream_t cudaStream) const;

private:
  std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> modToUnpDefault;
  bool hasQuality_;

  SiPixelFedCablingMapGPU *cablingMapHost = nullptr;  // pointer to struct in CPU

  struct GPUData {
    ~GPUData();
    SiPixelFedCablingMapGPU *cablingMapDevice = nullptr;  // pointer to struct in GPU
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    ~ModulesToUnpack();
    unsigned char *modToUnpDefault = nullptr;  // pointer to GPU
  };
  cms::cuda::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif
