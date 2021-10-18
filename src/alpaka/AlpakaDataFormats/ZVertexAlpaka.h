#ifndef AlpakaDataFormatsVertexZVertexHeterogeneous_H
#define AlpakaDataFormatsVertexZVertexHeterogeneous_H

#include "AlpakaDataFormats/ZVertexSoA.h"

#include "AlpakaCore/device_unique_ptr.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ZVertexAlpaka = cms::alpakatools::device::unique_ptr<ZVertexSoA>;
  using ZVertexHost = cms::alpakatools::host::unique_ptr<ZVertexSoA>;

  // NB: ANOTHER OPTION IS TO CREATE A HeterogeneousSoA class,
  // with a AlpakaDeviceBuf<ZVertexSoA> as a data member
  // and a toHostAsync function.

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
