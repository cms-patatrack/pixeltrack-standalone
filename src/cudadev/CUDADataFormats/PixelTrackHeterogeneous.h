#ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
#define CUDADataFormats_Track_PixelTrackHeterogeneous_h

#include "CUDADataFormats/HeterogeneousSoA.h"
#include "CUDADataFormats/TrackSoAHeterogeneousT.h"

using PixelTrackHeterogeneous = HeterogeneousSoA<pixelTrack::TrackSoA>;

#endif  // #ifndef CUDADataFormats_Track_PixelTrackHeterogeneous_h
