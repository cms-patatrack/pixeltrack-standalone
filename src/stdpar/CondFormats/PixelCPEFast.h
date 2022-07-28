#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>
#include <memory>

#include "CUDACore/ESProduct.h"
#include "CUDACore/HostAllocator.h"
#include "CondFormats/pixelCPEforGPU.h"

class PixelCPEFast {
public:
  PixelCPEFast(std::string const& path);

  ~PixelCPEFast();

  pixelCPEforGPU::ParamsOnGPU const* get() const { return m_params.get(); }

private:
  std::shared_ptr<pixelCPEforGPU::ParamsOnGPU> m_params;
  std::shared_ptr<pixelCPEforGPU::CommonParams> m_commonParams;
  std::shared_ptr<pixelCPEforGPU::DetParams[]> m_detParams;
  std::shared_ptr<pixelCPEforGPU::LayerGeometry> m_layerGeometry;
  std::shared_ptr<pixelCPEforGPU::AverageGeometry> m_averageGeometry;
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
