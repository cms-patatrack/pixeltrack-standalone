#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include <Kokkos_Core.hpp>

template <typename MemorySpace>
class SiPixelFedCablingMapGPUWrapper {
public:
  using CablingMapView = Kokkos::View<const SiPixelFedCablingMapGPU, MemorySpace>;

  explicit SiPixelFedCablingMapGPUWrapper(CablingMapView cablingMap, bool quality)
      : cablingMapDevice_{std::move(cablingMap)}, hasQuality_{quality} {}
  ~SiPixelFedCablingMapGPUWrapper() = default;

  bool hasQuality() const { return hasQuality_; }

  const CablingMapView& cablingMap() const { return cablingMapDevice_; }

private:
  CablingMapView cablingMapDevice_;
  bool hasQuality_;
};

#endif
