#ifndef AlpakaDataFormats_alpaka_PixelTrackAlpaka_h
#define AlpakaDataFormats_alpaka_PixelTrackAlpaka_h

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/PixelTrackSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using PixelTrackAlpaka = cms::alpakatools::device_buffer<Device, pixelTrack::TrackSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // AlpakaDataFormats_alpaka_PixelTrackAlpaka_h
