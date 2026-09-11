// Checks one set of digis and clusters against another, column by column.
//
// Two SiPixelDigisSoA and two SiPixelClustersSoA products in one event, told
// apart by the label of the module that produced them -- the C++
// SiPixelRawToClusterCUDA and the Python pixel_clusters.py.  Both decode the
// same FED buffers with the same integer arithmetic, so every column is
// expected to agree exactly, cluster numbering included: the clusters of a
// module are numbered in the order their first pixel appears, which is a
// property of the labelling rather than of the order the pixels are walked in.
//
// Only the entries the producers actually wrote are compared.  Both allocate
// their columns for the largest event the detector can produce and fill the
// first nDigis of them; the rest is untouched memory in either case.

#include <cstdint>
#include <iostream>
#include <string>

#include "CUDADataFormats/SiPixelClustersSoA.h"
#include "CUDADataFormats/SiPixelDigisSoA.h"
#include "CUDADataFormats/gpuClusteringConstants.h"
#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

class ClusterCompare : public edm::EDProducer {
public:
  ClusterCompare(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : lhsLabel_(config.required<std::string>("lhs")),
        rhsLabel_(config.required<std::string>("rhs")),
        lhsDigis_(reg.consumes<SiPixelDigisSoA>(lhsLabel_)),
        rhsDigis_(reg.consumes<SiPixelDigisSoA>(rhsLabel_)),
        lhsClusters_(reg.consumes<SiPixelClustersSoA>(lhsLabel_)),
        rhsClusters_(reg.consumes<SiPixelClustersSoA>(rhsLabel_)) {}

  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    auto const& lhsDigis = event.get(lhsDigis_);
    auto const& rhsDigis = event.get(rhsDigis_);
    auto const& lhsClusters = event.get(lhsClusters_);
    auto const& rhsClusters = event.get(rhsClusters_);

    ++events_;
    auto disagree = [&](std::string const& what, long lhs, long rhs) {
      ++differing_;
      std::cout << "ClusterCompare Event " << event.eventID() << ": " << what << " is " << lhs << " for "
                << lhsLabel_ << " and " << rhs << " for " << rhsLabel_ << std::endl;
    };

    if (lhsDigis.nDigis() != rhsDigis.nDigis()) {
      return disagree("N(digis)", lhsDigis.nDigis(), rhsDigis.nDigis());
    }
    if (lhsDigis.nModules() != rhsDigis.nModules()) {
      return disagree("N(modules)", lhsDigis.nModules(), rhsDigis.nModules());
    }
    if (lhsClusters.nClusters() != rhsClusters.nClusters()) {
      return disagree("N(clusters)", lhsClusters.nClusters(), rhsClusters.nClusters());
    }

    const uint32_t nDigis = lhsDigis.nDigis();
    const uint32_t nModules = lhsDigis.nModules();
    unsigned int wrong = 0;
    for (uint32_t i = 0; i < nDigis; ++i) {
      if (lhsDigis.c_xx()[i] != rhsDigis.c_xx()[i] or lhsDigis.c_yy()[i] != rhsDigis.c_yy()[i] or
          lhsDigis.c_adc()[i] != rhsDigis.c_adc()[i] or
          lhsDigis.c_moduleInd()[i] != rhsDigis.c_moduleInd()[i] or
          lhsDigis.c_clus()[i] != rhsDigis.c_clus()[i] or lhsDigis.c_pdigi()[i] != rhsDigis.c_pdigi()[i] or
          lhsDigis.c_rawIdArr()[i] != rhsDigis.c_rawIdArr()[i]) {
        ++wrong;
      }
    }
    digis_ += nDigis;

    unsigned int wrongModules = 0;
    for (uint32_t i = 0; i < nModules; ++i) {
      if (lhsClusters.c_moduleStart()[1 + i] != rhsClusters.c_moduleStart()[1 + i] or
          lhsClusters.c_moduleId()[i] != rhsClusters.c_moduleId()[i]) {
        ++wrongModules;
      }
    }
    for (uint32_t i = 0; i < gpuClustering::MaxNumModules; ++i) {
      if (lhsClusters.c_clusInModule()[i] != rhsClusters.c_clusInModule()[i] or
          lhsClusters.c_clusModuleStart()[i] != rhsClusters.c_clusModuleStart()[i]) {
        ++wrongModules;
      }
    }
    modules_ += nModules;

    if (wrong != 0 or wrongModules != 0) {
      ++differing_;
      wrongDigis_ += wrong;
      wrongModules_ += wrongModules;
      std::cout << "ClusterCompare Event " << event.eventID() << ": " << wrong << " of " << nDigis
                << " digis and " << wrongModules << " modules differ" << std::endl;
    }
  }

  void endJob() override {
    std::cout << "ClusterCompare " << events_ << " events, " << digis_ << " digis in " << modules_
              << " modules, " << differing_ << " events with any difference, " << wrongDigis_
              << " digis and " << wrongModules_ << " module entries differing" << std::endl;
  }

private:
  const std::string lhsLabel_;
  const std::string rhsLabel_;
  const edm::EDGetTokenT<SiPixelDigisSoA> lhsDigis_;
  const edm::EDGetTokenT<SiPixelDigisSoA> rhsDigis_;
  const edm::EDGetTokenT<SiPixelClustersSoA> lhsClusters_;
  const edm::EDGetTokenT<SiPixelClustersSoA> rhsClusters_;
  unsigned int events_ = 0;
  unsigned int differing_ = 0;
  unsigned long digis_ = 0;
  unsigned long modules_ = 0;
  unsigned long wrongDigis_ = 0;
  unsigned long wrongModules_ = 0;
};

DEFINE_FWK_MODULE(ClusterCompare);
