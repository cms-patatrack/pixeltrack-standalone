#include "AlpakaCore/alpakaDevAcc.h"
#include "AlpakaCore/initialise.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void initialise() { devices = enumerate<Device>(); }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
