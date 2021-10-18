#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include "AlpakaCore/device_unique_ptr.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelFedCablingMapGPUWrapper {
  public:
    using CablingMapDevicePtr = cms::alpakatools::device::unique_ptr<SiPixelFedCablingMapGPU>;

    explicit SiPixelFedCablingMapGPUWrapper(CablingMapDevicePtr cablingMap, bool quality)
        : cablingMapDevice_{std::move(cablingMap)}, hasQuality_{quality} {}
    ~SiPixelFedCablingMapGPUWrapper() = default;

    bool hasQuality() const { return hasQuality_; }

    const SiPixelFedCablingMapGPU* cablingMap() const { return cablingMapDevice_.get(); }

  private:
    CablingMapDevicePtr cablingMapDevice_;
    bool hasQuality_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
