#include <atomic>
#include <cassert>
#include <future>
#include <iostream>
#include <map>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventRange.h"
#include "Framework/PluginFactory.h"

namespace {
  std::atomic<int> nevents = 0;
}

// test using an std::map instead of a simpler std::vector
using TestBatchingProducerExternalWorkAsyncState = std::map<int, std::future<int>>;

class TestBatchingProducerExternalWork
    : public edm::EDBatchingProducerExternalWork<TestBatchingProducerExternalWorkAsyncState> {
public:
  explicit TestBatchingProducerExternalWork(edm::ProductRegistry& reg);

private:
  void acquire(edm::ConstEventRange events,
               edm::EventSetup const& eventSetup,
               edm::WaitingTaskWithArenaHolder holder,
               AsyncState& state) const override;
  void produce(edm::EventRange events, edm::EventSetup const& eventSetup, AsyncState& state) override;

  void endJob() override;

  const edm::EDGetTokenT<unsigned int> getToken_;
};

TestBatchingProducerExternalWork::TestBatchingProducerExternalWork(edm::ProductRegistry& reg)
    : getToken_(reg.consumes<unsigned int>()) {}

void TestBatchingProducerExternalWork::acquire(edm::ConstEventRange events,
                                               edm::EventSetup const& eventSetup,
                                               edm::WaitingTaskWithArenaHolder holder,
                                               AsyncState& state) const {
  for (edm::Event const& event : events) {
    auto const value = event.get(getToken_);
    assert(value == static_cast<unsigned int>(event.eventID() + 10 * event.streamID() + 100));

    // cannot move form the holder as it is used more than once
    state[event.eventID()] = std::async([holder]() mutable {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1s);
      holder.doneWaiting();
      return 42;
    });

#ifndef FWTEST_SILENT
    std::cout << "TestBatchingProducerExternalWork::acquire Event " << event.eventID() << " stream " << event.streamID()
              << " value " << value << std::endl;
#endif
  }
}

void TestBatchingProducerExternalWork::produce(edm::EventRange events,
                                               edm::EventSetup const& eventSetup,
                                               AsyncState& state) {
#ifndef FWTEST_SILENT
  for (edm::Event& event : events) {
    std::cout << "TestBatchingProducerExternalWork::produce Event " << event.eventID() << " stream " << event.streamID()
              << " from future " << state[event.eventID()].get() << std::endl;
  }
#endif
  ++nevents;
}

void TestBatchingProducerExternalWork::endJob() {
  std::cout << "TestBatchingProducerExternalWork::endJob processed " << nevents.load() << " events" << std::endl;
}

DEFINE_FWK_MODULE(TestBatchingProducerExternalWork);
