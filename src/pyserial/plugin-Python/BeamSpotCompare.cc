// Checks one BeamSpotPOD against another, field by field.
//
// Two products of the same type in one event, told apart by the label of the
// module that produced them -- the C++ BeamSpotToPOD and the Python
// beam_spot.py.  The two are copies of the same EventSetup product, so any
// difference at all is a bug: the comparison is exact.

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "DataFormats/BeamSpotPOD.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

namespace {
  struct Field {
    char const* name;
    float BeamSpotPOD::*member;
  };

  constexpr Field kFields[] = {{"x", &BeamSpotPOD::x},
                               {"y", &BeamSpotPOD::y},
                               {"z", &BeamSpotPOD::z},
                               {"sigmaZ", &BeamSpotPOD::sigmaZ},
                               {"beamWidthX", &BeamSpotPOD::beamWidthX},
                               {"beamWidthY", &BeamSpotPOD::beamWidthY},
                               {"dxdz", &BeamSpotPOD::dxdz},
                               {"dydz", &BeamSpotPOD::dydz},
                               {"emittanceX", &BeamSpotPOD::emittanceX},
                               {"emittanceY", &BeamSpotPOD::emittanceY},
                               {"betaStar", &BeamSpotPOD::betaStar}};
}  // namespace

class BeamSpotCompare : public edm::EDProducer {
public:
  BeamSpotCompare(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : lhsLabel_(config.required<std::string>("lhs")),
        rhsLabel_(config.required<std::string>("rhs")),
        lhsToken_(reg.consumes<BeamSpotPOD>(lhsLabel_)),
        rhsToken_(reg.consumes<BeamSpotPOD>(rhsLabel_)) {}

  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    auto const& lhs = event.get(lhsToken_);
    auto const& rhs = event.get(rhsToken_);

    std::ostringstream differences;
    for (auto const& field : kFields) {
      const float a = lhs.*(field.member);
      const float b = rhs.*(field.member);
      if (a != b) {
        differences << "  " << field.name << ": " << a << " != " << b << '\n';
      }
    }

    if (differences.tellp() != 0) {
      ++mismatches_;
      std::cout << "BeamSpotCompare Event " << event.eventID() << ": " << lhsLabel_ << " and " << rhsLabel_
                << " differ\n"
                << differences.str();
    }
    ++events_;
  }

  void endJob() override {
    std::cout << "BeamSpotCompare " << events_ << " events, " << mismatches_ << " with any difference between "
              << lhsLabel_ << " and " << rhsLabel_ << std::endl;
  }

private:
  const std::string lhsLabel_;
  const std::string rhsLabel_;
  const edm::EDGetTokenT<BeamSpotPOD> lhsToken_;
  const edm::EDGetTokenT<BeamSpotPOD> rhsToken_;
  unsigned int events_ = 0;
  unsigned int mismatches_ = 0;
};

DEFINE_FWK_MODULE(BeamSpotCompare);
