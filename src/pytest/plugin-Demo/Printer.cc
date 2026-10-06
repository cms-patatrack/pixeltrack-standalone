// Prints the comparison result, and optionally the first few values of a
// Floats product.  Adds nothing to the Event.

#include <cstddef>
#include <iostream>
#include <string>
#include <syncstream>

#include "DataFormats/Products.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class Printer : public edm::EDProducer {
public:
  Printer(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : label_(config.required<std::string>("input")),
        show_(config.optional<std::size_t>("show", 4)),
        comparison_(reg.consumes<pytest::Comparison>(label_)) {}

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    pytest::Comparison const& comparison = event.get(comparison_);

    // One osyncstream emits its whole contents atomically when destroyed, so a
    // concurrently running stream cannot interleave into the middle of a line.
    std::osyncstream out(std::cout);
    out << "Printer      Event " << event.eventID() << " stream " << event.streamID() << " compared "
        << comparison.size << " values: " << (comparison.agree() ? "agree" : "DISAGREE")
        << ", largest deviation " << comparison.maxDeviation;
    if (comparison.mismatches != 0) {
      out << ", " << comparison.mismatches << " mismatch(es)";
    }
    out << "\n";
  }

  const std::string label_;
  const std::size_t show_;
  const edm::EDGetTokenT<pytest::Comparison> comparison_;
};

DEFINE_FWK_MODULE(Printer);
