#ifndef CUDADataFormatsVertexZVertex_H
#define CUDADataFormatsVertexZVertex_H

#include <memory>

#include "CUDACore/Product.h"
#include "CUDADataFormats/ZVertexSoA.h"
#include "CUDADataFormats/PixelTrack.h"

using ZVertex = std::unique_ptr<ZVertexSoA>;
using ZVertexCUDAProduct = cms::cuda::Product<ZVertex>;

#endif
