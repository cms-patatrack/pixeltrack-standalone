#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "CUDADataFormats/ZVertexSoA.h"
#include "CUDADataFormats/HeterogeneousSoA.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;

#endif
