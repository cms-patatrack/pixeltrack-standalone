#ifndef Test3_AlpakaAlgoProducer_h
#define Test3_AlpakaAlgoProducer_h

#include "AlpakaCore/alpakaConfigFwd.h"
#include "AlpakaDataFormats/TestProduct.h"

#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class AlpakaAlgoProducer {
  public:
    AlpakaAlgoProducer();
    ~AlpakaAlgoProducer();

    TestProduct<Queue> run();

  private:
    std::unique_ptr<Device> device_; // demonstrate member depending (weakly) on Device
    std::shared_ptr<Queue> queue_; // demonstrate member depending (weakly) on Queue
  };
}

#endif
