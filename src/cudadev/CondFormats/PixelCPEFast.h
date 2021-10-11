#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CondFormats/pixelCPEforGPU.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

class PixelCPEFast {
public:
  PixelCPEFast(cms::cuda::device::unique_ptr<pixelCPEforGPU::ParamsOnGPU> gpuData,
               cms::cuda::device::unique_ptr<pixelCPEforGPU::CommonParams> commonParams,
               cms::cuda::device::unique_ptr<pixelCPEforGPU::DetParams[]> detParams,
               cms::cuda::device::unique_ptr<pixelCPEforGPU::LayerGeometry> layerGeometry,
               cms::cuda::device::unique_ptr<pixelCPEforGPU::AverageGeometry> averageGeometry)
      : gpuData_(std::move(gpuData)),
        commonParams_(std::move(commonParams)),
        detParams_(std::move(detParams)),
        layerGeometry_(std::move(layerGeometry)),
        averageGeometry_(std::move(averageGeometry)) {}

  const pixelCPEforGPU::ParamsOnGPU *params() const { return gpuData_.get(); }

private:
  cms::cuda::device::unique_ptr<pixelCPEforGPU::ParamsOnGPU> gpuData_;
  cms::cuda::device::unique_ptr<pixelCPEforGPU::CommonParams> commonParams_;
  cms::cuda::device::unique_ptr<pixelCPEforGPU::DetParams[]> detParams_;
  cms::cuda::device::unique_ptr<pixelCPEforGPU::LayerGeometry> layerGeometry_;
  cms::cuda::device::unique_ptr<pixelCPEforGPU::AverageGeometry> averageGeometry_;
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
