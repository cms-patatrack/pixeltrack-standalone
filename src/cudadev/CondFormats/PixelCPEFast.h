#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CUDACore/ESProduct.h"
#include "CUDACore/HostAllocator.h"
#include "CondFormats/pixelCPEforGPU.h"

class PixelCPEFast {
public:
  PixelCPEFast(std::string const &path);

  ~PixelCPEFast() = default;

  // The return value can only be used safely in kernels launched on
  // the same cudaStream, or after cudaStreamSynchronize.
  const pixelCPEforGPU::ParamsOnGPU *getGPUProductAsync(cudaStream_t cudaStream) const;

  pixelCPEforGPU::ParamsOnGPU const &getCPUProduct() const { return cpuData_; }

private:
  // allocate this with posix malloc to be compatible with the cpu workflow
  std::vector<pixelCPEforGPU::DetParams> detParamsGPU_;
  pixelCPEforGPU::CommonParams commonParamsGPU_;
  pixelCPEforGPU::LayerGeometry layerGeometry_;
  pixelCPEforGPU::AverageGeometry averageGeometry_;
  pixelCPEforGPU::ParamsOnGPU cpuData_;

  struct GPUData {
    ~GPUData();
    // not needed if not used on CPU...
    pixelCPEforGPU::ParamsOnGPU paramsOnGPU_h;
    pixelCPEforGPU::ParamsOnGPU *paramsOnGPU_d = nullptr;  // copy of the above on the Device
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  void fillParamsForGpu();
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
