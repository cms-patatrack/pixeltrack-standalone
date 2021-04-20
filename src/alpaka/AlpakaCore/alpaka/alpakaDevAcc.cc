#include "AlpakaCore/alpakaDevAcc.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  const DevAcc1 device = alpaka::getDevByIdx<PltfAcc1>(0u);
}
