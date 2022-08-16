#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <memory>

#include "CondFormats/pixelCPEforGPU.h"

class PixelCPEFast {
public:
  PixelCPEFast(std::string const& path);

  ~PixelCPEFast();

  pixelCPEforGPU::ParamsOnGPU const* get() const { return m_params.get(); }

private:
  std::unique_ptr<pixelCPEforGPU::ParamsOnGPU> m_params;
  std::unique_ptr<pixelCPEforGPU::CommonParams> m_commonParams;
  std::unique_ptr<pixelCPEforGPU::DetParams[]> m_detParams;
  std::unique_ptr<pixelCPEforGPU::LayerGeometry> m_layerGeometry;
  std::unique_ptr<pixelCPEforGPU::AverageGeometry> m_averageGeometry;
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
