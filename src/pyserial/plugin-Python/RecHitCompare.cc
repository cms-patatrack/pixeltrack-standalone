// Checks one set of pixel rec hits against another, column by column.
//
// Two TrackingRecHit2DCPU products in one event, told apart by the label of
// the module that produced them -- the C++ SiPixelRecHitCUDA and the Python
// pixel_rechits.py.  They are computed from the same clusters by the same
// arithmetic, so the floating-point columns are expected to agree exactly;
// the tolerance is there to say by how much they do not, rather than to
// excuse a difference.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class RecHitCompare : public edm::EDProducer {
public:
  RecHitCompare(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : lhsLabel_(config.required<std::string>("lhs")),
        rhsLabel_(config.required<std::string>("rhs")),
        tolerance_(config.optional<float>("tolerance", 0.0f)),
        lhsToken_(reg.consumes<TrackingRecHit2DCPU>(lhsLabel_)),
        rhsToken_(reg.consumes<TrackingRecHit2DCPU>(rhsLabel_)) {}

  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    auto const& lhs = *event.get(lhsToken_).view();
    auto const& rhs = *event.get(rhsToken_).view();

    ++events_;
    if (lhs.nHits() != rhs.nHits()) {
      ++differing_;
      std::cout << "RecHitCompare Event " << event.eventID() << ": " << lhs.nHits() << " hits against "
                << rhs.nHits() << std::endl;
      return;
    }

    unsigned int wrong = 0;
    for (uint32_t i = 0, n = lhs.nHits(); i < n; ++i) {
      bool same = lhs.detectorIndex(i) == rhs.detectorIndex(i) and lhs.charge(i) == rhs.charge(i) and
                  lhs.clusterSizeX(i) == rhs.clusterSizeX(i) and lhs.clusterSizeY(i) == rhs.clusterSizeY(i) and
                  lhs.iphi(i) == rhs.iphi(i);
      const float differences[] = {std::abs(lhs.xLocal(i) - rhs.xLocal(i)),
                                   std::abs(lhs.yLocal(i) - rhs.yLocal(i)),
                                   std::abs(lhs.xerrLocal(i) - rhs.xerrLocal(i)),
                                   std::abs(lhs.yerrLocal(i) - rhs.yerrLocal(i)),
                                   std::abs(lhs.xGlobal(i) - rhs.xGlobal(i)),
                                   std::abs(lhs.yGlobal(i) - rhs.yGlobal(i)),
                                   std::abs(lhs.zGlobal(i) - rhs.zGlobal(i)),
                                   std::abs(lhs.rGlobal(i) - rhs.rGlobal(i))};
      for (const float difference : differences) {
        worst_ = std::max(worst_, difference);
        if (difference > tolerance_) {
          same = false;
        }
      }
      if (not same) {
        ++wrong;
      }
    }

    hits_ += lhs.nHits();
    if (wrong != 0) {
      ++differing_;
      wrongHits_ += wrong;
      std::cout << "RecHitCompare Event " << event.eventID() << ": " << wrong << " of " << lhs.nHits()
                << " hits differ" << std::endl;
    }
  }

  void endJob() override {
    std::cout << "RecHitCompare " << events_ << " events, " << hits_ << " hits, " << differing_
              << " events with any difference, " << wrongHits_ << " hits differing; worst |difference| = " << worst_
              << std::endl;
  }

private:
  const std::string lhsLabel_;
  const std::string rhsLabel_;
  const float tolerance_;
  const edm::EDGetTokenT<TrackingRecHit2DCPU> lhsToken_;
  const edm::EDGetTokenT<TrackingRecHit2DCPU> rhsToken_;
  unsigned int events_ = 0;
  unsigned int differing_ = 0;
  unsigned long hits_ = 0;
  unsigned long wrongHits_ = 0;
  float worst_ = 0;
};

DEFINE_FWK_MODULE(RecHitCompare);
