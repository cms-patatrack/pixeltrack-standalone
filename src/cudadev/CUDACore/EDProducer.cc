#include "CUDACore/EDProducer.h"

namespace cms::cuda {
  void EDProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    runProduce(event.streamID(), [&](auto& ctx) { produce(event, eventSetup, ctx); });
  }

  void SynchronizingEDProducer::acquire(edm::Event const& event,
                                        edm::EventSetup const& eventSetup,
                                        edm::WaitingTaskWithArenaHolder holder) {
    runAcquire(event.streamID(), std::move(holder), [&](auto& ctx) { acquire(event, eventSetup, ctx); });
  }

  void SynchronizingEDProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    runProduce(event.streamID(), [&](auto& ctx) { produce(event, eventSetup, ctx); });
  }
}  // namespace cms::cuda
