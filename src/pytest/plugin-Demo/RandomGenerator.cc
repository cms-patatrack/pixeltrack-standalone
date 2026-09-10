// Publishes a vector of random floats in [-1, 1].
//
// The event index is added to the seed, so every event sees different data
// while the whole job stays reproducible from one `seed`.  It is read from the
// Event rather than captured in the constructor because this instance is
// reused for every event its stream processes.

#include <cstddef>
#include <cstdint>
#include <random>

#include "DataFormats/Products.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class RandomGenerator : public edm::EDProducer {
public:
  RandomGenerator(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : count_(config.optional<std::size_t>("count", 1000)),
        seed_(config.optional<std::uint64_t>("seed", 12345)),
        token_(reg.produces<pytest::Floats>()) {}

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    std::mt19937_64 engine(seed_ + static_cast<std::uint64_t>(event.eventID()));
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    pytest::Floats values(count_);
    for (float& value : values) {
      value = distribution(engine);
    }
    event.emplace(token_, std::move(values));
  }

  const std::size_t count_;
  const std::uint64_t seed_;
  const edm::EDPutTokenT<pytest::Floats> token_;
};

DEFINE_FWK_MODULE(RandomGenerator);
