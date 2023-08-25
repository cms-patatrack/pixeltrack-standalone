#ifndef AlpakaDataFormats_PixelTrackHost_h
#define AlpakaDataFormats_PixelTrackHost_h

#include "AlpakaCore/config.h"
#include "AlpakaCore/memory.h"
#include "AlpakaDataFormats/PixelTrackSoA.h"

using PixelTrackHost = cms::alpakatools::host_buffer<pixelTrack::TrackSoA>;

#endif  // AlpakaDataFormats_PixelTrackHost_h
