#ifndef PixelVertexFinding_kokkos_gpuClusterFillHist_h
#define PixelVertexFinding_kokkos_gpuClusterFillHist_h

#include "KokkosCore/kokkos_assert.h"
#include "KokkosCore/HistoContainer.h"

#include "gpuVertexFinder.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {
    // this algo does not really scale as it works in a single block...
    // enough for <10K tracks we have
    template <typename Histo>
    KOKKOS_INLINE_FUNCTION void clusterFillHist(Kokkos::View<ZVertices, KokkosExecSpace> vdata,
                                                Kokkos::View<WorkSpace, KokkosExecSpace> vws,
                                                Kokkos::View<Histo*, KokkosExecSpace> vhist,
                                                int minT,       // min number of neighbours to be "core"
                                                float eps,      // max absolute distance to cluster
                                                float errmax,   // max error to be "seed"
                                                float chi2max,  // max normalized distance to cluster
                                                const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false

      const auto leagueRank = team_member.league_rank();
      auto id = leagueRank * team_member.team_size() + team_member.team_rank();

      if (verbose && 0 == id)
        printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);

      auto& __restrict__ data = *vdata.data();
      auto& __restrict__ ws = *vws.data();

      auto nt = ws.ntrks;
      float const* __restrict__ zt = ws.zt;

      uint8_t* __restrict__ izt = ws.izt;
      int32_t* __restrict__ nn = data.ndof;
      int32_t* __restrict__ iv = ws.iv;

      assert(vdata.data());
      assert(zt);

      auto* localHist = &vhist(leagueRank);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, Histo::totbins()),
                           [=](int j) { localHist->off[j] = 0; });
      team_member.team_barrier();

      if (verbose && 0 == id)
        printf("booked hist with %d bins, size %d for %d tracks\n", localHist->nbins(), localHist->capacity(), nt);

      assert(nt <= localHist->capacity());

      // fill hist  (bin shall be wider than "eps")
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
        assert(i < static_cast<int>(ZVertices::MAXTRACKS));
        int iz = int(zt[i] * 10.);  // valid if eps<=0.1
        // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
        iz = std::min(std::max(iz, INT8_MIN), INT8_MAX);
        izt[i] = iz - INT8_MIN;
        assert(iz - INT8_MIN >= 0);
        assert(iz - INT8_MIN < 256);
        localHist->count(izt[i]);
        iv[i] = i;
        nn[i] = 0;
      });
      team_member.team_barrier();
    }
  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_kokkos_gpuClusterFillHist_h
