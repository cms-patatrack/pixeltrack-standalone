#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CUDACore/ESProduct.h"
#include "CUDACore/HostAllocator.h"
#include "CondFormats/pixelCPEforGPU.h"

class PixelCPEFast {
public:
  PixelCPEFast(std::string const& path);

  ~PixelCPEFast();

  pixelCPEforGPU::ParamsOnGPU const* get() const { return m_params; }

private:
  pixelCPEforGPU::ParamsOnGPU* m_params = nullptr;
  pixelCPEforGPU::CommonParams* m_commonParams = nullptr;
  pixelCPEforGPU::DetParams* m_detParams = nullptr;
  pixelCPEforGPU::LayerGeometry* m_layerGeometry = nullptr;
  pixelCPEforGPU::AverageGeometry* m_averageGeometry = nullptr;
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
