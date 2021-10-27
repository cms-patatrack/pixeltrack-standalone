#ifndef HeterogeneousCore_AlpakaCore_chooseDevice_h
#define HeterogeneousCore_AlpakaCore_chooseDevice_h

#include "AlpakaCore/alpakaConfig.h"
#include "Framework/Event.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  ::ALPAKA_ACCELERATOR_NAMESPACE::Device const& chooseDevice(edm::StreamID id);

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaCore_chooseDevice_h
