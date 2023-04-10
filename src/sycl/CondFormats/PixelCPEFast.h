#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "SYCLCore/ESProduct.h"
#include "CondFormats/pixelCPEforGPU.h"

class PixelCPEFast {
public:
  PixelCPEFast(std::string const &path);

  ~PixelCPEFast() = default;

  // The return value can only be used safely in kernels launched on
  // the same queue, or after queue.wait()
  const pixelCPEforGPU::ParamsOnGPU *getGPUProductAsync(sycl::queue stream) const;

private:
  // allocate it with posix malloc to be compatible with cpu wf
  std::vector<pixelCPEforGPU::DetParams> m_detParamsGPU;
  pixelCPEforGPU::CommonParams m_commonParamsGPU;
  pixelCPEforGPU::LayerGeometry m_layerGeometry;
  pixelCPEforGPU::AverageGeometry m_averageGeometry;

  struct GPUData {
    ~GPUData();
    // not needed if not used on CPU...
    pixelCPEforGPU::ParamsOnGPU h_paramsOnGPU;
    // copy of the above on the Device
    cms::sycltools::device::unique_ptr<pixelCPEforGPU::ParamsOnGPU> d_paramsOnGPU = nullptr;
  };
  cms::sycltools::ESProduct<GPUData> gpuData_;

  void fillParamsForGpu();
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
