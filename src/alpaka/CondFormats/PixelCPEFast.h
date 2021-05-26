#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CondFormats/pixelCPEforGPU.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelCPEFast {
  public:
    PixelCPEFast(AlpakaDeviceBuf<pixelCPEforGPU::CommonParams> commonParams,
                 AlpakaDeviceBuf<pixelCPEforGPU::DetParams> detParams,
                 AlpakaDeviceBuf<pixelCPEforGPU::LayerGeometry> layerGeometry,
                 AlpakaDeviceBuf<pixelCPEforGPU::AverageGeometry> averageGeometry,
                 AlpakaDeviceBuf<pixelCPEforGPU::ParamsOnGPU> params)
        : m_commonParams(std::move(commonParams)),
          m_detParams(std::move(detParams)),
          m_layerGeometry(std::move(layerGeometry)),
          m_averageGeometry(std::move(averageGeometry)),
          m_params(std::move(params)) {}

    ~PixelCPEFast() = default;

    pixelCPEforGPU::ParamsOnGPU const* params() const { return alpaka::getPtrNative(m_params); }

  private:
    AlpakaDeviceBuf<pixelCPEforGPU::CommonParams> m_commonParams;
    AlpakaDeviceBuf<pixelCPEforGPU::DetParams> m_detParams;
    AlpakaDeviceBuf<pixelCPEforGPU::LayerGeometry> m_layerGeometry;
    AlpakaDeviceBuf<pixelCPEforGPU::AverageGeometry> m_averageGeometry;
    AlpakaDeviceBuf<pixelCPEforGPU::ParamsOnGPU> m_params;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
