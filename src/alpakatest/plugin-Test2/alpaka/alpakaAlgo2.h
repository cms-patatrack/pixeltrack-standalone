#ifndef Test2_alpakaAlgo2_h
#define Test2_alpakaAlgo2_h

#include "AlpakaCore/alpakaConfigFwd.h"

#include <any>

// In principle this could be ALPAKA_ARCHITECTURE_NAMESPACE, but how
// to compile it only once for CPU? Current build rules build separate
// objects for serial and TBB accelerators. Do we need to have one set
// of build rules for files to be compiled by accelerator, and
// different set for files to be compiled by architecture?
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  std::any alpakaAlgo2(Queue& queue);
}

#endif
