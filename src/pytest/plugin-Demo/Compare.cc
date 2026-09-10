// Compares two Floats products element by element and publishes the result.
//
// This is what makes the C++/Python comparison a check rather than an
// eyeball: with `tolerance = 0` the two implementations have to agree bit for
// bit, and squaring is a single correctly-rounded IEEE multiply on both sides,
// so they can.  Both inputs are the same C++ type and are told apart only by
// the label of the module that produced them.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "DataFormats/Products.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class Compare : public edm::EDProducer {
public:
  Compare(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : tolerance_(config.optional<float>("tolerance", 0.0f)),
        strict_(config.optional<bool>("strict", true)),
        lhsLabel_(config.required<std::string>("lhs")),
        rhsLabel_(config.required<std::string>("rhs")),
        lhs_(reg.consumes<pytest::Floats>(lhsLabel_)),
        rhs_(reg.consumes<pytest::Floats>(rhsLabel_)),
        output_(reg.produces<pytest::Comparison>()) {
    if (not(tolerance_ >= 0.0f)) {
      throw std::runtime_error("Compare: 'tolerance' must be non-negative");
    }
  }

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    pytest::Floats const& lhs = event.get(lhs_);
    pytest::Floats const& rhs = event.get(rhs_);

    pytest::Comparison result;
    result.tolerance = tolerance_;

    if (lhs.size() != rhs.size()) {
      throw std::runtime_error("Compare: '" + lhsLabel_ + "' has " + std::to_string(lhs.size()) +
                               " elements but '" + rhsLabel_ + "' has " + std::to_string(rhs.size()));
    }

    result.size = lhs.size();
    for (std::size_t i = 0; i < lhs.size(); ++i) {
      // A NaN on one side only is a mismatch; two NaNs in the same place are
      // the two implementations agreeing.
      if (std::isnan(lhs[i]) or std::isnan(rhs[i])) {
        if (std::isnan(lhs[i]) != std::isnan(rhs[i])) {
          ++result.mismatches;
        }
        continue;
      }
      const float deviation = std::abs(lhs[i] - rhs[i]);
      if (deviation > result.maxDeviation) {
        result.maxDeviation = deviation;
      }
      if (deviation > tolerance_) {
        ++result.mismatches;
      }
    }

    const bool agree = result.agree();
    const std::size_t mismatches = result.mismatches;
    const float maxDeviation = result.maxDeviation;
    event.emplace(output_, std::move(result));

    if (strict_ and not agree) {
      throw std::runtime_error("Compare: '" + lhsLabel_ + "' and '" + rhsLabel_ + "' disagree in " +
                               std::to_string(mismatches) + " of " + std::to_string(lhs.size()) +
                               " elements, largest deviation " + std::to_string(maxDeviation));
    }
  }

  const float tolerance_;
  const bool strict_;
  const std::string lhsLabel_;
  const std::string rhsLabel_;
  const edm::EDGetTokenT<pytest::Floats> lhs_;
  const edm::EDGetTokenT<pytest::Floats> rhs_;
  const edm::EDPutTokenT<pytest::Comparison> output_;
};

DEFINE_FWK_MODULE(Compare);
