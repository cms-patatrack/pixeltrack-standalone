#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "AlpakaDataFormats/ZVertexSoA.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ZVertexAlpaka = AlpakaDeviceBuf<ZVertexSoA>;
  using ZVertexHost = AlpakaHostBuf<ZVertexSoA>;

  // NB: ANOTHER OPTION IS TO CREATE A HeterogeneousSoA class,
  // with a AlpakaDeviceBuf<ZVertexSoA> as a data member
  // and a toHostAsync function.

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
