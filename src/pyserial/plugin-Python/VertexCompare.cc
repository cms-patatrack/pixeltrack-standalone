// Compares two ZVertexHeterogeneous products, produced by two different
// modules from the same tracks.
//
// This is the check that the Python vertex finder reproduces the C++ one, and
// it is only expressible because products are keyed by (type, label): both
// modules publish a ZVertexHeterogeneous, and this one names which is which.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <syncstream>
#include <vector>

#include "CUDADataFormats/ZVertexHeterogeneous.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class VertexCompare : public edm::EDProducer {
public:
  VertexCompare(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : lhsLabel_(config.required<std::string>("lhs")),
        rhsLabel_(config.required<std::string>("rhs")),
        tolerance_(config.optional<float>("tolerance", 1e-4f)),
        lhs_(reg.consumes<ZVertexHeterogeneous>(lhsLabel_)),
        rhs_(reg.consumes<ZVertexHeterogeneous>(rhsLabel_)) {}

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    auto const& a = *event.get(lhs_).get();
    auto const& b = *event.get(rhs_).get();

    // Vertices are found in whatever order the clustering happens to produce,
    // so the comparison is on the sorted z positions rather than index by
    // index.
    auto zs = [](ZVertexSoA const& soa) {
      std::vector<float> out(soa.zv, soa.zv + soa.nvFinal);
      std::sort(out.begin(), out.end());
      return out;
    };
    auto za = zs(a);
    auto zb = zs(b);

    float worst = 0.f;
    const std::size_t common = std::min(za.size(), zb.size());
    for (std::size_t i = 0; i < common; ++i) {
      worst = std::max(worst, std::abs(za[i] - zb[i]));
    }

    std::osyncstream out(std::cout);
    out << "VertexCompare Event " << event.eventID() << ": " << lhsLabel_ << " n=" << a.nvFinal << "  "
        << rhsLabel_ << " n=" << b.nvFinal;
    if (a.nvFinal == b.nvFinal) {
      out << "  (equal)  worst |dz| = " << worst;
    } else {
      out << "  DIFFER by " << (int(b.nvFinal) - int(a.nvFinal));
    }
    out << "\n";
  }

  const std::string lhsLabel_;
  const std::string rhsLabel_;
  const float tolerance_;
  const edm::EDGetTokenT<ZVertexHeterogeneous> lhs_;
  const edm::EDGetTokenT<ZVertexHeterogeneous> rhs_;
};

DEFINE_FWK_MODULE(VertexCompare);
