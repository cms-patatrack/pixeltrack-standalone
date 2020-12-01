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

  template <typename T = void>
  class EDProducerExternalWork {
  public:
    using AsyncState = T;

    EDProducerExternalWork() = default;
    virtual ~EDProducerExternalWork() = default;

    bool hasAcquire() const { return true; }

    void doAcquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {
      acquire(event, eventSetup, std::move(holder), state_);
    }

    virtual void acquire(Event const& event,
                         EventSetup const& eventSetup,
                         WaitingTaskWithArenaHolder holder,
                         AsyncState& state) const = 0;

    void doProduce(Event& event, EventSetup const& eventSetup) {
      produce(event, eventSetup, state_);
    }

    virtual void produce(Event& event, EventSetup const& eventSetup, AsyncState& state) = 0;

    void doEndJob() { endJob(); }
    virtual void endJob() {}

  private:
    AsyncState state_;
  };

  template <>
  class EDProducerExternalWork<void> {
  public:
    EDProducerExternalWork() = default;
    virtual ~EDProducerExternalWork() = default;

    bool hasAcquire() const { return true; }

    void doAcquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {
      acquire(event, eventSetup, std::move(holder));
    }

    void doProduce(Event& event, EventSetup const& eventSetup) { produce(event, eventSetup); }

    virtual void acquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) const = 0;
    virtual void produce(Event& event, EventSetup const& eventSetup) = 0;

    void doEndJob() { endJob(); }
    virtual void endJob() {}

  private:
  };
}  // namespace edm

#endif
