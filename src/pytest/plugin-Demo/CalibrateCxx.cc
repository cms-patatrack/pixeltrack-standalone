// Builds a Samples product from a Floats product and the calibration.
//
// The C++ half of the binding comparison: python/calibrate.py builds the same
// product through the generated bindings, and SamplesCompare checks that the
// two agree column by column.  Neither the work nor the product means anything
// physically -- what it stands in for is the *shape* of pyserial's products,
// which no test in this backend can carry.
//
// `input` is optional: an empty label means "the only Floats product", which
// is what every module here wants until a configuration runs two producers of
// one type.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "DataFormats/Products.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

class CalibrateCxx : public edm::EDProducer {
public:
  CalibrateCxx(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : input_(reg.consumes<pytest::Floats>(config.optional<std::string>("input", ""))),
        output_(reg.produces<pytest::Samples>()) {}

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    pytest::Floats const& input = event.get(input_);
    pytest::Calibration const& calibration = eventSetup.get<pytest::Calibration>();

    const std::uint32_t n = static_cast<std::uint32_t>(input.size());
    std::vector<std::int32_t> channels(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      channels[i] = static_cast<std::int32_t>(i) % calibration.channels();
    }

    pytest::Samples samples(n, calibration, channels);
    pytest::Samples::View& view = samples.view();
    std::copy(input.begin(), input.end(), view.valueSpan().begin());
    view.filled = n;
    samples.applyCalibration();

    event.emplace(output_, std::move(samples));
  }

  const edm::EDGetTokenT<pytest::Floats> input_;
  const edm::EDPutTokenT<pytest::Samples> output_;
};

DEFINE_FWK_MODULE(CalibrateCxx);
