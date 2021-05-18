#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelFedCablingMapGPUWrapper {
  public:
    using CablingMapDeviceBuf = AlpakaDeviceBuf<SiPixelFedCablingMapGPU>;

    explicit SiPixelFedCablingMapGPUWrapper(CablingMapDeviceBuf cablingMap, bool quality)
        : cablingMapDevice_{std::move(cablingMap)}, hasQuality_{quality} {}
    ~SiPixelFedCablingMapGPUWrapper() = default;

    bool hasQuality() const { return hasQuality_; }

    const SiPixelFedCablingMapGPU* cablingMap() const { return alpaka::getPtrNative(cablingMapDevice_); }

  private:
    CablingMapDeviceBuf cablingMapDevice_;
    bool hasQuality_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
