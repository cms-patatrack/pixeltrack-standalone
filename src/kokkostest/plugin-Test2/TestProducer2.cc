#include <cassert>
#include <future>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class TestProducer2 : public edm::EDProducerExternalWork {
public:
  explicit TestProducer2(edm::ProductRegistry& reg);

private:
  void acquire(edm::Event const& event,
               edm::EventSetup const& eventSetup,
               edm::WaitingTaskWithArenaHolder holder) override;
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

  edm::EDGetTokenT<unsigned int> getToken_;
  std::future<int> future_;
};

TestProducer2::TestProducer2(edm::ProductRegistry& reg) : getToken_(reg.consumes<unsigned int>()) {}

void TestProducer2::acquire(edm::Event const& event,
                            edm::EventSetup const& eventSetup,
                            edm::WaitingTaskWithArenaHolder holder) {
  auto const value = event.get(getToken_);
  assert(value == static_cast<unsigned int>(event.eventID() + 10 * event.streamID() + 100));

  future_ = std::async([holder = std::move(holder)]() mutable {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s);
    holder.doneWaiting(std::exception_ptr());
    return 42;
  });

  std::cout << "TestProducer2::acquire Event " << event.eventID() << " stream " << event.streamID() << " value "
            << value << std::endl;
}

void TestProducer2::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  std::cout << "TestProducer2::produce Event " << event.eventID() << " stream " << event.streamID() << " from future "
            << future_.get() << std::endl;
}

DEFINE_FWK_MODULE(TestProducer2);
