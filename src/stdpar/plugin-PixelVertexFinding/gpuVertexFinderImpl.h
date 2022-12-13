#include <algorithm>
#include <atomic>
#include <execution>
#include <ranges>
#include <memory>
#include "CUDACore/cudaCheck.h"

#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

namespace gpuVertexFinder {

  void loadTracks(TkSoA const* ptracks, ZVertexSoA* soa, WorkSpace* pws, float ptMin) {
    assert(ptracks);
    assert(soa);
    auto const& tracks = *ptracks;
    auto const& fit = tracks.stateAtBS;
    auto const* quality = tracks.qualityData();

    auto iter{std::views::iota(0, TkSoA::stride())};
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
      auto nHits = tracks.nHits(idx);
      if (nHits == 0)
        return;  // this is a guard: maybe we need to move to nTracks...

      // initialize soa...
      soa->idv[idx] = -1;

      if (nHits < 4)
        return;  // no triplets
      if (quality[idx] != trackQuality::loose)
        return;

      auto pt = tracks.pt(idx);

      if (pt < ptMin)
        return;

      auto& data = *pws;
      std::atomic_ref<uint32_t> inc{data.ntrks};
      auto it = inc++;
      data.itrk[it] = idx;
      data.zt[it] = tracks.zip(idx);
      data.ezt2[it] = fit.covariance(idx)(14);
      data.ptt2[it] = pt * pt;
    });
  }

  ZVertex Producer::makeAsync(TkSoA const* tksoa, float ptMin) const {
    // std::cout << "producing Vertices on GPU" << std::endl;
    ZVertex vertices{std::make_unique<ZVertexSoA>()};

    assert(tksoa);
    auto* soa = vertices.get();
    assert(soa);
    auto ws = std::make_unique<WorkSpace>();

    init(soa, ws.get());
    loadTracks(tksoa, soa, ws.get(), ptMin);
    if (useDensity_ || oneKernel_) {
      clusterTracksByDensity(soa, ws.get(), minT, eps, errmax, chi2max);
    } else if (useDBSCAN_) {
      clusterTracksDBSCAN(soa, ws.get(), minT, eps, errmax, chi2max);
    } else if (useIterative_) {
      clusterTracksIterative(soa, ws.get(), minT, eps, errmax, chi2max);
    }
    fitVertices(soa, ws.get(), 50.);
    splitVertices(soa, ws.get(), 9.f);
    fitVertices(soa, ws.get(), 5000.);

    sortByPt2(soa, ws.get());

    return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
