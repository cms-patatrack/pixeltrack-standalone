#ifndef HeterogeneousCore_AlpakaUtilities_SharedEventPtr_h
#define HeterogeneousCore_AlpakaUtilities_SharedEventPtr_h

#include <memory>
#include <type_traits>

#include "AlpakaCore/alpakaConfig.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  using SharedEventPtr = std::shared_ptr<::ALPAKA_ACCELERATOR_NAMESPACE::Event>;

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif
