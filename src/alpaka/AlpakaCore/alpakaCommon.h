#ifndef ALPAKACOMMON_H
#define ALPAKACOMMON_H

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemoryHelper.h"
//#include "AlpakaCore/alpakaWorkDivHelper.h"


static const DevHost host = alpaka::getDevByIdx<PltfHost>(0u);

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  static const DevAcc1 device = alpaka::getDevByIdx<PltfAcc1>(0u);
}


#endif  // ALPAKACOMMON_H
