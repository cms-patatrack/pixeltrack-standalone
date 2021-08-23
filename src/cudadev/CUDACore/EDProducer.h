#ifndef HeterogeneousCore_CUDACore_stream_EDProducer_h
#define HeterogeneousCore_CUDACore_stream_EDProducer_h

#include "Framework/EDProducer.h"
#include "CUDACore/Context.h"

namespace cms::cuda {
  class EDProducer : public edm::EDProducer {
  public:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
    virtual void produce(edm::Event& event, edm::EventSetup const& eventSetup, ProduceContext& context) = 0;
  };

  class SynchronizingEDProducer : public edm::EDProducerExternalWork {
  public:
    void acquire(edm::Event const& event,
                 edm::EventSetup const& eventSetup,
                 edm::WaitingTaskWithArenaHolder holder) override;
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    virtual void acquire(edm::Event const& event, edm::EventSetup const& eventSetup, AcquireContext& context) = 0;
    virtual void produce(edm::Event& event, edm::EventSetup const& eventSetup, ProduceContext& context) = 0;

  private:
    ContextState state_;
  };
}  // namespace cms::cuda

#endif
