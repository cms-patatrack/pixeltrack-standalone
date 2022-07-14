#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "CUDADataFormats/ZVertexSoA.h"
#include "CUDADataFormats/HeterogeneousSoA.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"

#ifdef CUDAUVM_DISABLE_MANAGED_VERTEX
using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
#else
using ZVertexHeterogeneous = ManagedSoA<ZVertexSoA>;
#endif
#ifndef __CUDACC__
#include "CUDACore/Product.h"
using ZVertexCUDAProduct = cms::cuda::Product<ZVertexHeterogeneous>;
#endif

#endif
