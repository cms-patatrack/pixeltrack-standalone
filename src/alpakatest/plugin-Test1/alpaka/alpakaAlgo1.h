#ifndef Test1_alpakaAlgo1_h
#define Test1_alpakaAlgo1_h

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"

// In principle this could be ALPAKA_ARCHITECTURE_NAMESPACE, but how
// to compile it only once for CPU? Current build rules build separate
// objects for serial and TBB accelerators. Do we need to have one set
// of build rules for files to be compiled by accelerator, and
// different set for files to be compiled by architecture?
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  cms::alpakatools::device_buffer<Device, float[]> alpakaAlgo1();
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
