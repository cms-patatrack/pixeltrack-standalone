#ifndef Test1_alpakaAlgo1_h
#define Test1_alpakaAlgo1_h

#include "AlpakaCore/alpakaConfig.h"

// In principle this could be ALPAKA_ARCHITECTURE_NAMESPACE, but how
// to compile it only once for CPU? Current build rules build separate
// objects for serial and TBB accelerators. Do we need to have one set
// of build rules for files to be compiled by accelerator, and
// different set for files to be compiled by architecture?
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  //alpaka::mem::buf::Buf<Acc2, float, std::integral_constant<long unsigned int, 1>, unsigned int> alpakaAlgo1();
  alpaka::mem::buf::Buf<Acc2, float, Dim2, Idx> alpakaAlgo1();
  //float* alpakaAlgo1();

}

#endif
