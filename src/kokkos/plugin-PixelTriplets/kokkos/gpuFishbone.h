#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "Geometry/phase1PixelTopology.h"
#include "KokkosCore/VecArray.h"
#include "KokkosCore/kokkos_assert.h"
#include "KokkosCore/kokkosConfig.h"
#include "KokkosDataFormats/approx_atan2.h"

#include "../GPUCACell.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuPixelDoublets {
    KOKKOS_INLINE_FUNCTION void fishbone(TrackingRecHit2DSOAView const* __restrict__ hhp,
                                         Kokkos::View<GPUCACell*, KokkosExecSpace> cells,
                                         Kokkos::View<uint32_t, KokkosExecSpace> nCells,  // not used
                                         Kokkos::View<GPUCACell::OuterHitOfCell*, KokkosExecSpace> isOuterHitOfCell,
                                         uint32_t nHits,
                                         bool checkTrack,
                                         const uint32_t stride,
                                         const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& teamMember) {
      constexpr auto maxCellsPerHit = GPUCACell::maxCellsPerHit;

      auto const& hh = *hhp;
      // auto layer = [&](uint16_t id) { return hh.cpeParams().layer(id); };

      const int teamRank = teamMember.team_rank();
      const int teamSize = teamMember.team_size();
      const int leagueRank = teamMember.league_rank();
      const int leagueSize = teamMember.league_size();

      // x run faster...
      const uint32_t blockDim = teamSize / stride;
      uint32_t firstX = teamRank % stride;
      uint32_t firstY = leagueRank * blockDim + teamRank / stride;

      float x[maxCellsPerHit], y[maxCellsPerHit], z[maxCellsPerHit], n[maxCellsPerHit];
      uint16_t d[maxCellsPerHit];  // uint8_t l[maxCellsPerHit];
      uint32_t cc[maxCellsPerHit];

      for (int idy = firstY, nt = nHits; idy < nt; idy += leagueSize * blockDim) {
        auto const& vc = isOuterHitOfCell(idy);
        auto s = vc.size();
        if (s < 2)
          continue;
        // if alligned kill one of the two.
        // in principle one could try to relax the cut (only in r-z?) for jumping-doublets
        auto const& c0 = cells(vc[0]);
        auto xo = c0.get_outer_x(hh);
        auto yo = c0.get_outer_y(hh);
        auto zo = c0.get_outer_z(hh);
        auto sg = 0;
        for (int32_t ic = 0; ic < s; ++ic) {
          auto& ci = cells(vc[ic]);
          if (0 == ci.theUsed)
            continue;  // for triplets equivalent to next
          if (checkTrack && ci.tracks().empty())
            continue;
          cc[sg] = vc[ic];
          d[sg] = ci.get_inner_detIndex(hh);
          //      l[sg] = layer(d[sg]);
          x[sg] = ci.get_inner_x(hh) - xo;
          y[sg] = ci.get_inner_y(hh) - yo;
          z[sg] = ci.get_inner_z(hh) - zo;
          n[sg] = x[sg] * x[sg] + y[sg] * y[sg] + z[sg] * z[sg];
          ++sg;
        }
        if (sg < 2)
          continue;
        // here we parallelize
        for (int32_t ic = firstX; ic < sg - 1; ic += stride) {
          auto& ci = cells(cc[ic]);
          for (auto jc = ic + 1; jc < sg; ++jc) {
            auto& cj = cells(cc[jc]);
            // must be different detectors (in the same layer)
            //        if (d[ic]==d[jc]) continue;
            // || l[ic]!=l[jc]) continue;
            auto cos12 = x[ic] * x[jc] + y[ic] * y[jc] + z[ic] * z[jc];
            if (d[ic] != d[jc] && cos12 * cos12 >= 0.99999f * n[ic] * n[jc]) {
              // alligned:  kill farthest  (prefer consecutive layers)
              if (n[ic] > n[jc]) {
                ci.theDoubletId = -1;
                break;
              } else {
                cj.theDoubletId = -1;
              }
            }
          }  //cj
        }    // ci
      }      // hits
    }
  }  // namespace gpuPixelDoublets
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h
