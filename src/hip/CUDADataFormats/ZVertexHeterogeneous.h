#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "CUDADataFormats/ZVertexSoA.h"
#include "CUDADataFormats/HeterogeneousSoA.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
#ifndef __HIPCC__
#include "CUDACore/Product.h"
using ZVertexCUDAProduct = cms::hip::Product<ZVertexHeterogeneous>;
#endif

#endif
