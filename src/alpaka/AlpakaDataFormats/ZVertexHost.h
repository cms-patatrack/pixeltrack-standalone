#ifndef AlpakaDataFormats_ZVertexHost_h
#define AlpakaDataFormats_ZVertexHost_h

#include "AlpakaCore/config.h"
#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/ZVertexSoA.h"

using ZVertexHost = cms::alpakatools::host_buffer<ZVertexSoA>;

#endif  // AlpakaDataFormats_ZVertexHost_h
