#ifndef CUDADataFormatsVertexZVertex_H
#define CUDADataFormatsVertexZVertex_H

#include <memory>

#include "CUDADataFormats/ZVertexSoA.h"
#include "CUDADataFormats/PixelTrack.h"

using ZVertex = std::unique_ptr<ZVertexSoA>;

#endif
