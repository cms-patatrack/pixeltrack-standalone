#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include <memory>

#include "CUDADataFormats/ZVertexSoA.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = std::unique_ptr<ZVertexSoA>;

#include "CUDACore/Product.h"
using ZVertexCUDAProduct = cms::cuda::Product<ZVertexHeterogeneous>;


#endif
