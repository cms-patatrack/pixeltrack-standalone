// Publishes the element-wise square of the Floats product named by `input`.
//
// The C++ half of the comparison: python/square.py does the same arithmetic
// with numpy, and Compare checks that the two agree.

#include <string>

#include "DataFormats/Products.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class SquareCxx : public edm::EDProducer {
public:
  SquareCxx(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : input_(reg.consumes<pytest::Floats>(config.required<std::string>("input"))),
        output_(reg.produces<pytest::Floats>()) {}

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    pytest::Floats const& input = event.get(input_);

    pytest::Floats squared(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
      squared[i] = input[i] * input[i];
    }
    event.emplace(output_, std::move(squared));
  }

  const edm::EDGetTokenT<pytest::Floats> input_;
  const edm::EDPutTokenT<pytest::Floats> output_;
};

DEFINE_FWK_MODULE(SquareCxx);
