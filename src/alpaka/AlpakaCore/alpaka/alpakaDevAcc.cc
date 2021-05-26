#include "AlpakaCore/alpakaDevAcc.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  const DevAcc2 device = alpaka::getDevByIdx<PltfAcc2>(0u);
}
