#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CondFormats/pixelCPEforGPU.h"

#include <Kokkos_Core.hpp>

template <typename MemorySpace>
class PixelCPEFast {
public:
  PixelCPEFast(Kokkos::View<pixelCPEforGPU::CommonParams, MemorySpace> commonParams,
               Kokkos::View<pixelCPEforGPU::DetParams*, MemorySpace> detParams,
               Kokkos::View<pixelCPEforGPU::LayerGeometry, MemorySpace> layerGeometry,
               Kokkos::View<pixelCPEforGPU::AverageGeometry, MemorySpace> averageGeometry,
               Kokkos::View<pixelCPEforGPU::ParamsOnGPU, MemorySpace> params)
      : m_commonParams(std::move(commonParams)),
        m_detParams(std::move(detParams)),
        m_layerGeometry(std::move(layerGeometry)),
        m_averageGeometry(std::move(averageGeometry)),
        m_params(std::move(params)) {}

  ~PixelCPEFast() = default;

  Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, MemorySpace> const& params() const { return m_params; }

private:
  Kokkos::View<pixelCPEforGPU::CommonParams const, MemorySpace> m_commonParams;
  Kokkos::View<pixelCPEforGPU::DetParams const*, MemorySpace> m_detParams;
  Kokkos::View<pixelCPEforGPU::LayerGeometry const, MemorySpace> m_layerGeometry;
  Kokkos::View<pixelCPEforGPU::AverageGeometry const, MemorySpace> m_averageGeometry;
  Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, MemorySpace> m_params;
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
