#ifndef Test3_AlpakaAlgoConsumer_h
#define Test3_AlpakaAlgoConsumer_h

#include "AlpakaCore/alpakaConfigFwd.h"
#include "AlpakaDataFormats/TestProduct.h"

#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class AlpakaAlgoConsumer {
  public:
    AlpakaAlgoConsumer() = default;
    ~AlpakaAlgoConsumer() = default;

    void run(const TestProduct<Queue>& input);
  };
}

#endif
