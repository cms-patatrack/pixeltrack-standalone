// Compares two Samples products column by column.
//
// Every column the Python module wrote through a zero-copy view is checked
// against the one the C++ module wrote directly, with tolerance = 0: a binding
// that hands Python a copy instead of a view, or a view of the wrong length,
// shows up here as a difference rather than as a plausible number.
//
// `count` is the sample count the Python module published as a bare int, so
// the scalar round trip is checked too.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "DataFormats/Products.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class SamplesCompare : public edm::EDProducer {
public:
  SamplesCompare(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : lhsLabel_(config.required<std::string>("lhs")),
        rhsLabel_(config.required<std::string>("rhs")),
        countLabel_(config.optional<std::string>("count", "")),
        lhs_(reg.consumes<pytest::Samples>(lhsLabel_)),
        rhs_(reg.consumes<pytest::Samples>(rhsLabel_)),
        count_(reg.consumes<int>(countLabel_)),
        output_(reg.produces<pytest::Comparison>()) {}

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    pytest::Samples const& lhs = event.get(lhs_);
    pytest::Samples const& rhs = event.get(rhs_);
    const int count = event.get(count_);

    if (lhs.size() != rhs.size()) {
      throw std::runtime_error("SamplesCompare: '" + lhsLabel_ + "' has " + std::to_string(lhs.size()) +
                               " samples but '" + rhsLabel_ + "' has " + std::to_string(rhs.size()));
    }
    if (static_cast<std::uint32_t>(count) != rhs.size()) {
      throw std::runtime_error("SamplesCompare: '" + countLabel_ + "' published " + std::to_string(count) +
                               " but '" + rhsLabel_ + "' has " + std::to_string(rhs.size()) + " samples");
    }
    if (lhs.view().filled != rhs.view().filled) {
      // A view bound by value rather than by reference loses this write.
      throw std::runtime_error("SamplesCompare: '" + lhsLabel_ + "' filled " +
                               std::to_string(lhs.view().filled) + " but '" + rhsLabel_ + "' filled " +
                               std::to_string(rhs.view().filled));
    }

    pytest::Comparison result;
    result.size = lhs.size();

    auto const& left = lhs.view();
    auto const& right = rhs.view();
    const auto compare = [&](auto&& a, auto&& b) {
      for (std::size_t i = 0; i < a.size(); ++i) {
        const float deviation = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (deviation > result.maxDeviation) {
          result.maxDeviation = deviation;
        }
        if (deviation > 0.0f) {
          ++result.mismatches;
        }
      }
    };
    // The views are const, so these are the const spans.
    compare(left.valueSpan(), right.valueSpan());
    compare(left.scaledSpan(), right.scaledSpan());
    compare(left.channelSpan(), right.channelSpan());

    const bool agree = result.agree();
    const std::size_t mismatches = result.mismatches;
    const float maxDeviation = result.maxDeviation;
    event.emplace(output_, std::move(result));

    if (not agree) {
      throw std::runtime_error("SamplesCompare: '" + lhsLabel_ + "' and '" + rhsLabel_ + "' disagree in " +
                               std::to_string(mismatches) + " of " + std::to_string(3 * lhs.size()) +
                               " values, largest deviation " + std::to_string(maxDeviation));
    }
  }

  const std::string lhsLabel_;
  const std::string rhsLabel_;
  const std::string countLabel_;
  const edm::EDGetTokenT<pytest::Samples> lhs_;
  const edm::EDGetTokenT<pytest::Samples> rhs_;
  const edm::EDGetTokenT<int> count_;
  const edm::EDPutTokenT<pytest::Comparison> output_;
};

DEFINE_FWK_MODULE(SamplesCompare);
