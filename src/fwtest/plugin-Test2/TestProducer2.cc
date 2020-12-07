#include <atomic>
#include <cassert>
#include <future>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

namespace {
  std::atomic<int> nevents = 0;
}

class TestProducer2 : public edm::EDProducerExternalWork<std::future<int>> {
public:
  explicit TestProducer2(edm::ProductRegistry& reg);

private:
  void acquire(edm::Event const& event,
               edm::EventSetup const& eventSetup,
               edm::WaitingTaskWithArenaHolder holder,
               AsyncState& state) const override;
  void produce(edm::Event& event, edm::EventSetup const& eventSetup, AsyncState& state) override;

  void endJob() override;

  const edm::EDGetTokenT<unsigned int> getToken_;
};

TestProducer2::TestProducer2(edm::ProductRegistry& reg) : getToken_(reg.consumes<unsigned int>()) {}

void TestProducer2::acquire(edm::Event const& event,
                            edm::EventSetup const& eventSetup,
                            edm::WaitingTaskWithArenaHolder holder,
                            AsyncState& state) const {
  auto const value = event.get(getToken_);
  assert(value == static_cast<unsigned int>(event.eventID() + 10 * event.streamID() + 100));

  state = std::async([holder = std::move(holder)]() mutable {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s);
    holder.doneWaiting();
    return 42;
  });

#ifndef FWTEST_SILENT
  std::cout << "TestProducer2::acquire Event " << event.eventID() << " stream " << event.streamID() << " value "
            << value << std::endl;
#endif
}

void TestProducer2::produce(edm::Event& event, edm::EventSetup const& eventSetup, AsyncState& state) {
#ifndef FWTEST_SILENT
  std::cout << "TestProducer2::produce Event " << event.eventID() << " stream " << event.streamID() << " from future "
            << state.get() << std::endl;
#endif
  ++nevents;
}

void TestProducer2::endJob() {
  std::cout << "TestProducer2::endJob processed " << nevents.load() << " events" << std::endl;
}

DEFINE_FWK_MODULE(TestProducer2);
