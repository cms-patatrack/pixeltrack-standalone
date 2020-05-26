#ifndef EDProducerBase_h
#define EDProducerBase_h

#include "Framework/WaitingTaskWithArenaHolder.h"

namespace edm {
  class Event;
  class EventSetup;

  class EDProducer {
  public:
    EDProducer() = default;
    virtual ~EDProducer() = default;

    bool hasAcquire() const { return false; }

    void doAcquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {}

    void doProduce(Event& event, EventSetup const& eventSetup) { produce(event, eventSetup); }

    virtual void produce(Event& event, EventSetup const& eventSetup) = 0;

    void doEndJob() { endJob(); }

    virtual void endJob() {}

  private:
  };

  class EDProducerExternalWork {
  public:
    EDProducerExternalWork() = default;
    virtual ~EDProducerExternalWork() = default;

    bool hasAcquire() const { return true; }

    void doAcquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {
      acquire(event, eventSetup, std::move(holder));
    }

    void doProduce(Event& event, EventSetup const& eventSetup) { produce(event, eventSetup); }

    virtual void acquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) = 0;
    virtual void produce(Event& event, EventSetup const& eventSetup) = 0;

    void doEndJob() { endJob(); }
    virtual void endJob() {}

  private:
  };
}  // namespace edm

#endif
