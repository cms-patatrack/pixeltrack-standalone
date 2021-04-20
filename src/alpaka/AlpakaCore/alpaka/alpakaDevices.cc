#include "AlpakaCore/alpakaDevices.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  const DevHost host = alpaka::getDevByIdx<PltfHost>(0u);
  const DevAcc1 device = alpaka::getDevByIdx<PltfAcc1>(0u);
}
