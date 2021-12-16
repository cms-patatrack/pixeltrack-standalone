#ifndef Test1_alpakaAlgo1_h
#define Test1_alpakaAlgo1_h

#include "AlpakaCore/alpakaConfigFwd.h"
#include "AlpakaCore/alpakaMemory.h"

#include <any>

// In principle this could be ALPAKA_ARCHITECTURE_NAMESPACE, but how
// to compile it only once for CPU? Current build rules build separate
// objects for serial and TBB accelerators. Do we need to have one set
// of build rules for files to be compiled by accelerator, and
// different set for files to be compiled by architecture?
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  std::any alpakaAlgo1(Queue& queue);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
