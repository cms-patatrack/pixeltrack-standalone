#ifndef AlpakaDataFormats_ZVertexAlpaka_h
#define AlpakaDataFormats_ZVertexAlpaka_h

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/ZVertexSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ZVertexAlpaka = cms::alpakatools::device_buffer<Device, ZVertexSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

using ZVertexHost = cms::alpakatools::host_buffer<ZVertexSoA>;

#endif  // AlpakaDataFormats_ZVertexAlpaka_h
