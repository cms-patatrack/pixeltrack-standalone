#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>
#include <vector>

#include "CondFormats/pixelCPEforGPU.h"

class PixelCPEFast {
public:
  PixelCPEFast(std::string const &path);

  ~PixelCPEFast() = default;

  pixelCPEforGPU::ParamsOnGPU const &getCPUProduct() const { return cpuData_; }

private:
  // allocate it with posix malloc to be ocmpatible with cpu wf
  std::vector<pixelCPEforGPU::DetParams> m_detParamsGPU;
  // std::vector<pixelCPEforGPU::DetParams, cms::cuda::HostAllocator<pixelCPEforGPU::DetParams>> m_detParamsGPU;
  pixelCPEforGPU::CommonParams m_commonParamsGPU;
  pixelCPEforGPU::LayerGeometry m_layerGeometry;
  pixelCPEforGPU::AverageGeometry m_averageGeometry;

  pixelCPEforGPU::ParamsOnGPU cpuData_;
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
