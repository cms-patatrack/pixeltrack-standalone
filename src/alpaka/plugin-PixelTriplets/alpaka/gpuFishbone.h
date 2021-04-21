#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "AlpakaCore/alpakaKernelCommon.h"

#include "DataFormats/approx_atan2.h"
#include "Geometry/phase1PixelTopology.h"
#include "AlpakaCore/VecArray.h"

#include "GPUCACell.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace gpuPixelDoublets {

    //  __device__
    //  __forceinline__
    struct fishbone {
      template <typename T_Acc>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                    GPUCACell::Hits const* __restrict__ hhp,
                                    GPUCACell* cells,
                                    uint32_t const* __restrict__ nCells,
                                    GPUCACell::OuterHitOfCell const* __restrict__ isOuterHitOfCell,
                                    uint32_t nHits,
                                    bool checkTrack) const {
        constexpr auto maxCellsPerHit = GPUCACell::maxCellsPerHit;

        auto const& hh = *hhp;
        // auto layer = [&](uint16_t id) { return hh.cpeParams().layer(id); };

        float x[maxCellsPerHit], y[maxCellsPerHit], z[maxCellsPerHit], n[maxCellsPerHit];
        uint16_t d[maxCellsPerHit];  // uint8_t l[maxCellsPerHit];
        uint32_t cc[maxCellsPerHit];

        // X run faster...
        const uint32_t dimIndexY = 0u;
        const uint32_t dimIndexX = 1u;
        const uint32_t blockDimensionX(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndexX]);
        const auto& [firstElementIdxNoStrideX, endElementIdxNoStrideX] =
            cms::alpakatools::element_index_range_in_block(acc, 0u, dimIndexX);

        // Outermost loop on Y
        const uint32_t gridDimensionY(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[dimIndexY]);
        const auto& [firstElementIdxNoStrideY, endElementIdxNoStrideY] =
            cms::alpakatools::element_index_range_in_grid(acc, 0u, dimIndexY);
        uint32_t firstElementIdxY = firstElementIdxNoStrideY;
        uint32_t endElementIdxY = endElementIdxNoStrideY;
        for (uint32_t idy = firstElementIdxY, nt = nHits; idy < nt; ++idy) {
          if (!cms::alpakatools::next_valid_element_index_strided(
                  idy, firstElementIdxY, endElementIdxY, gridDimensionY, nt))
            break;

          auto const& vc = isOuterHitOfCell[idy];
          auto s = vc.size();
          if (s < 2)
            continue;
          // if alligned kill one of the two.
          // in principle one could try to relax the cut (only in r-z?) for jumping-doublets
          auto const& c0 = cells[vc[0]];
          auto xo = c0.get_outer_x(hh);
          auto yo = c0.get_outer_y(hh);
          auto zo = c0.get_outer_z(hh);
          uint32_t sg = 0;
          for (int32_t ic = 0; ic < s; ++ic) {
            auto& ci = cells[vc[ic]];
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
          // Here we parallelize in X
          uint32_t firstElementIdxX = firstElementIdxNoStrideX;
          uint32_t endElementIdxX = endElementIdxNoStrideX;
          for (uint32_t ic = firstElementIdxX; ic < sg - 1; ++ic) {
            if (!cms::alpakatools::next_valid_element_index_strided(
                    ic, firstElementIdxX, endElementIdxX, blockDimensionX, sg - 1))
              break;

            auto& ci = cells[cc[ic]];
            for (auto jc = ic + 1; jc < sg; ++jc) {
              auto& cj = cells[cc[jc]];
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

      }  // fishbone kernel operator ()
    };

  }  // namespace gpuPixelDoublets
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h
