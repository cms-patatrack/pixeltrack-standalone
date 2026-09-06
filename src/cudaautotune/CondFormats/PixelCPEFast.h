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
  // allocate it with posix malloc to be ocmpatible with cpu wf
  std::vector<pixelCPEforGPU::DetParams> m_detParamsGPU;
  // std::vector<pixelCPEforGPU::DetParams, cms::cuda::HostAllocator<pixelCPEforGPU::DetParams>> m_detParamsGPU;
  pixelCPEforGPU::CommonParams m_commonParamsGPU;
  pixelCPEforGPU::LayerGeometry m_layerGeometry;
  pixelCPEforGPU::AverageGeometry m_averageGeometry;

  pixelCPEforGPU::ParamsOnGPU cpuData_;

  struct GPUData {
    ~GPUData();
    // not needed if not used on CPU...
    pixelCPEforGPU::ParamsOnGPU h_paramsOnGPU;
    pixelCPEforGPU::ParamsOnGPU *d_paramsOnGPU = nullptr;  // copy of the above on the Device
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  void fillParamsForGpu();
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
