#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include <memory>

#include "CUDACore/Product.h"
#include "CUDADataFormats/ZVertexSoA.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = std::unique_ptr<ZVertexSoA>;
using ZVertexCUDAProduct = cms::cuda::Product<ZVertexHeterogeneous>;

#endif
