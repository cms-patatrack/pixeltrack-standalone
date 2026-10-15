#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "CUDADataFormats/ZVertexSoA.h"
#include "CUDADataFormats/HeterogeneousSoA.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
#ifndef __CUDACC__
#include "CUDACore/Product.h"
using ZVertexCUDAProduct = cms::cuda::Product<ZVertexHeterogeneous>;
#endif

#endif
