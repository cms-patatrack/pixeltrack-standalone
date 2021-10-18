#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CondFormats/pixelCPEforGPU.h"

#include "AlpakaCore/device_unique_ptr.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelCPEFast {
  public:
    PixelCPEFast(cms::alpakatools::device::unique_ptr<pixelCPEforGPU::CommonParams> commonParams,
                 cms::alpakatools::device::unique_ptr<pixelCPEforGPU::DetParams> detParams,
                 cms::alpakatools::device::unique_ptr<pixelCPEforGPU::LayerGeometry> layerGeometry,
                 cms::alpakatools::device::unique_ptr<pixelCPEforGPU::AverageGeometry> averageGeometry,
                 cms::alpakatools::device::unique_ptr<pixelCPEforGPU::ParamsOnGPU> params)
        : m_commonParams(std::move(commonParams)),
          m_detParams(std::move(detParams)),
          m_layerGeometry(std::move(layerGeometry)),
          m_averageGeometry(std::move(averageGeometry)),
          m_params(std::move(params)) {}

    ~PixelCPEFast() = default;

    pixelCPEforGPU::ParamsOnGPU const* params() const { return m_params.get(); }

  private:
    cms::alpakatools::device::unique_ptr<pixelCPEforGPU::CommonParams> m_commonParams;
    cms::alpakatools::device::unique_ptr<pixelCPEforGPU::DetParams> m_detParams;
    cms::alpakatools::device::unique_ptr<pixelCPEforGPU::LayerGeometry> m_layerGeometry;
    cms::alpakatools::device::unique_ptr<pixelCPEforGPU::AverageGeometry> m_averageGeometry;
    cms::alpakatools::device::unique_ptr<pixelCPEforGPU::ParamsOnGPU> m_params;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
