#include "AlpakaCore/alpakaDevAcc.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  const Device device = alpaka::getDevByIdx<Platform>(0u);
}
