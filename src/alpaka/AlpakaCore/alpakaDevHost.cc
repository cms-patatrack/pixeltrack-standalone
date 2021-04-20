#include "AlpakaCore/alpakaDevHost.h"

namespace alpaka_common {
  const DevHost host = alpaka::getDevByIdx<PltfHost>(0u);
}
