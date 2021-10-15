#ifndef HeterogeneousCore_AlpakaUtilities_SharedStreamPtr_h
#define HeterogeneousCore_AlpakaUtilities_SharedStreamPtr_h

#include <memory>
#include <type_traits>

#include "AlpakaCore/alpakaConfig.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  using SharedStreamPtr = std::shared_ptr<::ALPAKA_ACCELERATOR_NAMESPACE::Queue>;

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif
