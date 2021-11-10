#ifndef Test3_AlpakaAlgoIsolatedMember_h
#define Test3_AlpakaAlgoIsolatedMember_h

#include "AlpakaCore/alpakaConfigFwd.h"

#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class AlpakaAlgoIsolatedMember {
  public:
    AlpakaAlgoIsolatedMember();
    ~AlpakaAlgoIsolatedMember();
    void run();

  private:
    // e.g. std::optional does not work, it's declaration requires the definition of the type
    std::unique_ptr<Device> device_; // demonstrate member depending (weakly) on Device
    std::unique_ptr<Queue> queue_; // demonstrate member depending (weakly) on Queue
    int memberInt_;
  };
}

#endif
