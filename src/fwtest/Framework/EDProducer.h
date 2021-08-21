#ifndef EDProducerBase_h
#define EDProducerBase_h

#include <memory>

#include "Framework/EventRange.h"
#include "Framework/WaitingTaskWithArenaHolder.h"

namespace edm {
  class Event;
  class EventSetup;

  class EDProducer {
  public:
    EDProducer() = default;
    virtual ~EDProducer() = default;

    bool hasAcquire() const { return false; }

    void doAcquire(ConstEventRange events, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {}

    void doProduce(EventRange events, EventSetup const& eventSetup) {
      for (Event& event : events) {
        produce(event, eventSetup);
      }
    }

    virtual void produce(Event& event, EventSetup const& eventSetup) = 0;

    void doEndJob() { endJob(); }

    virtual void endJob() {}

  private:
  };

  class EDBatchingProducer {
  public:
    EDBatchingProducer() = default;
    virtual ~EDBatchingProducer() = default;

    bool hasAcquire() const { return false; }

    void doAcquire(ConstEventRange events, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {}

    void doProduce(EventRange events, EventSetup const& eventSetup) { produce(events, eventSetup); }

    virtual void produce(EventRange events, EventSetup const& eventSetup) = 0;

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

    void doAcquire(ConstEventRange events, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {
      if (events.size() > statesSize_) {
        statesSize_ = events.size();
        states_ = std::make_unique<AsyncState[]>(statesSize_);
      }
      for (size_t i = 0; i < events.size(); ++i) {
        acquire(events[i], eventSetup, holder, states_[i]);
      }
    }

    virtual void acquire(Event const& event,
                         EventSetup const& eventSetup,
                         WaitingTaskWithArenaHolder holder,
                         AsyncState& state) const = 0;

    void doProduce(EventRange events, EventSetup const& eventSetup) {
      for (size_t i = 0; i < events.size(); ++i) {
        produce(events[i], eventSetup, states_[i]);
      }
    }

    virtual void produce(Event& event, EventSetup const& eventSetup, AsyncState& state) = 0;

    void doEndJob() { endJob(); }

    virtual void endJob() {}

  private:
    size_t statesSize_ = 0;
    std::unique_ptr<AsyncState[]> states_;
  };

  template <>
  class EDProducerExternalWork<void> {
  public:
    EDProducerExternalWork() = default;
    virtual ~EDProducerExternalWork() = default;

    bool hasAcquire() const { return true; }

    void doAcquire(ConstEventRange events, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {
      for (Event const& event : events) {
        acquire(event, eventSetup, holder);
      }
    }

    virtual void acquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) const = 0;

    void doProduce(EventRange events, EventSetup const& eventSetup) {
      for (Event& event : events) {
        produce(event, eventSetup);
      }
    }

    virtual void produce(Event& event, EventSetup const& eventSetup) = 0;

    void doEndJob() { endJob(); }

    virtual void endJob() {}

  private:
  };

  template <typename T = void>
  class EDBatchingProducerExternalWork {
  public:
    using AsyncState = T;

    EDBatchingProducerExternalWork() = default;
    virtual ~EDBatchingProducerExternalWork() = default;

    bool hasAcquire() const { return true; }

    void doAcquire(ConstEventRange events, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {
      acquire(events, eventSetup, holder, state_);
    }

    virtual void acquire(ConstEventRange events,
                         EventSetup const& eventSetup,
                         WaitingTaskWithArenaHolder holder,
                         AsyncState& state) const = 0;

    void doProduce(EventRange events, EventSetup const& eventSetup) { produce(events, eventSetup, state_); }

    virtual void produce(EventRange events, EventSetup const& eventSetup, AsyncState& states) = 0;

    void doEndJob() { endJob(); }

    virtual void endJob() {}

  private:
    AsyncState state_;
  };

  template <>
  class EDBatchingProducerExternalWork<void> {
  public:
    EDBatchingProducerExternalWork() = default;
    virtual ~EDBatchingProducerExternalWork() = default;

    bool hasAcquire() const { return true; }

    void doAcquire(ConstEventRange events, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {
      acquire(events, eventSetup, holder);
    }

    virtual void acquire(ConstEventRange events,
                         EventSetup const& eventSetup,
                         WaitingTaskWithArenaHolder holder) const = 0;

    void doProduce(EventRange events, EventSetup const& eventSetup) { produce(events, eventSetup); }

    virtual void produce(EventRange events, EventSetup const& eventSetup) = 0;

    void doEndJob() { endJob(); }

    virtual void endJob() {}

  private:
  };

}  // namespace edm

#endif
