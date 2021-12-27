#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "AlpakaDataFormats/ZVertexSoA.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using ZVertexAlpaka = ::cms::alpakatools::device_buffer<Device, ZVertexSoA>;
  using ZVertexHost = ::cms::alpakatools::host_buffer<ZVertexSoA>;

  // NB: ANOTHER OPTION IS TO CREATE A HeterogeneousSoA class,
  // with a ::cms::alpakatools::device_buffer<Device, ZVertexSoA> as a data member
  // and a toHostAsync function.

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
