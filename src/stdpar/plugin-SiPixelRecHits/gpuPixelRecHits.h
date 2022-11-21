#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <limits>
#include <memory>
#include <ranges>

#include "CUDADataFormats/BeamSpot.h"
#include "CUDADataFormats/TrackingRecHit2D.h"
#include "DataFormats/approx_atan2.h"
#include "CUDACore/portableAtomicOp.h"
#include "CondFormats/pixelCPEforGPU.h"

namespace gpuPixelRecHits {

  void getHits(pixelCPEforGPU::ParamsOnGPU const* __restrict__ cpeParams,
               BeamSpotPOD const* __restrict__ bs,
               SiPixelDigis::DeviceConstView const* __restrict__ pdigis,
               int numElements,
               SiPixelClusters::DeviceConstView const* __restrict__ pclusters,
               TrackingRecHit2DSOAView* phits,
               uint32_t nModules) {
    // FIXME
    // the compiler seems NOT to optimize loads from views (even in a simple test case)
    // The whole gimnastic here of copying or not is a pure heuristic exercise that seems to produce the fastest code with the above signature
    // not using views (passing a gazzilion of array pointers) seems to produce the fastest code (but it is harder to mantain)

    assert(phits);
    assert(cpeParams);
    assert(nModules > 0);

    // to be moved in common namespace...
    constexpr uint16_t InvId = 9999;  // must be > MaxNumModules
    constexpr int32_t MaxHitsInIter = pixelCPEforGPU::MaxHitsInIter;
    using ClusParams = pixelCPEforGPU::ClusParams;

    // as usual one block per module
    auto clusParams{std::make_unique<ClusParams[]>(nModules)};
    ClusParams* clusParams_d{clusParams.get()};

    // copy average geometry corrected by beamspot . FIXME (move it somewhere else???)
    auto iter{std::views::iota(0u, TrackingRecHit2DSOAView::AverageGeometry::numberOfLaddersInBarrel)};
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto il) {
      auto& agc = phits->averageGeometry();
      auto const& ag = cpeParams->averageGeometry();
      agc.ladderZ[il] = ag.ladderZ[il] - bs->z;
      agc.ladderX[il] = ag.ladderX[il] - bs->x;
      agc.ladderY[il] = ag.ladderY[il] - bs->y;
      agc.ladderR[il] = sqrt(agc.ladderX[il] * agc.ladderX[il] + agc.ladderY[il] * agc.ladderY[il]);
      agc.ladderMinZ[il] = ag.ladderMinZ[il] - bs->z;
      agc.ladderMaxZ[il] = ag.ladderMaxZ[il] - bs->z;
    });
    auto& agc = phits->averageGeometry();
    auto const& ag = cpeParams->averageGeometry();
    agc.endCapZ[0] = ag.endCapZ[0] - bs->z;
    agc.endCapZ[1] = ag.endCapZ[1] - bs->z;
    //         printf("endcapZ %f %f\n",agc.endCapZ[0],agc.endCapZ[1]);
    auto iterModules{std::views::iota(0u, nModules)};
    std::for_each(
        std::execution::par,
        std::ranges::cbegin(iterModules),
        std::ranges::cend(iterModules),
        [=](const auto moduleIdx) {
          auto const& clusters = *pclusters;
          auto me = clusters.moduleId(moduleIdx);
          int nclus = clusters.clusInModule(me);
          auto& clusterParamsRef = clusParams_d[moduleIdx];
          if (0 == nclus)
            return;
          auto& hits = *phits;

          auto const digis = *pdigis;  // the copy is intentional!
#ifdef GPU_DEBUG
          auto k = clusters.moduleStart(1 + moduleIdx);
          while (digis.moduleInd(k) == InvId)
            ++k;
          assert(digis.moduleInd(k) == me);
#endif

#ifdef GPU_DEBUG
          if (me % 100 == 1)
            printf("hitbuilder: %d clusters in module %d. will write at %d\n", nclus, me, clusters.clusModuleStart(me));
#endif

          for (int startClus = 0, endClus = nclus; startClus < endClus; startClus += MaxHitsInIter) {
            auto first = clusters.moduleStart(1 + moduleIdx);

            int nClusInIter = std::min(MaxHitsInIter, endClus - startClus);
            int lastClus = startClus + nClusInIter;
            assert(nClusInIter <= nclus);
            assert(nClusInIter > 0);
            assert(lastClus <= nclus);

            assert(nclus > MaxHitsInIter || (0 == startClus && nClusInIter == nclus && lastClus == nclus));

            // init
            std::fill(clusterParamsRef.minRow, clusterParamsRef.minRow + nClusInIter, std::numeric_limits<uint32_t>::max());
            std::fill(clusterParamsRef.minCol, clusterParamsRef.minCol + nClusInIter, std::numeric_limits<uint32_t>::max());
            std::fill(clusterParamsRef.maxRow, clusterParamsRef.maxRow + nClusInIter, 0);
            std::fill(clusterParamsRef.maxCol, clusterParamsRef.maxCol + nClusInIter, 0);
            std::fill(clusterParamsRef.charge, clusterParamsRef.charge + nClusInIter, 0);
            std::fill(clusterParamsRef.Q_f_X, clusterParamsRef.Q_f_X + nClusInIter, 0);
            std::fill(clusterParamsRef.Q_l_X, clusterParamsRef.Q_l_X + nClusInIter, 0);
            std::fill(clusterParamsRef.Q_f_Y, clusterParamsRef.Q_f_Y + nClusInIter, 0);
            std::fill(clusterParamsRef.Q_l_Y, clusterParamsRef.Q_l_Y + nClusInIter, 0);

            // one thead per "digi"

            for (int i = first; i < numElements; ++i) {
              auto id = digis.moduleInd(i);
              if (id == InvId)
                continue;  // not valid
              if (id != me)
                break;  // end of module
              auto cl = digis.clus(i);
              if (cl < startClus || cl >= lastClus)
                continue;
              uint32_t x = digis.xx(i);
              uint32_t y = digis.yy(i);
              cl -= startClus;
              assert(cl >= 0);
              assert(cl < MaxHitsInIter);

              clusterParamsRef.minRow[cl] = std::min(clusterParamsRef.minRow[cl], x);
              clusterParamsRef.maxRow[cl] = std::max(clusterParamsRef.maxRow[cl], x);
              clusterParamsRef.minCol[cl] = std::min(clusterParamsRef.minCol[cl], x);
              clusterParamsRef.maxCol[cl] = std::max(clusterParamsRef.maxCol[cl], x);
            }

            // pixmx is not available in the binary dumps
            //auto pixmx = cpeParams->detParams(me).pixmx;
            auto pixmx = std::numeric_limits<uint16_t>::max();
            for (int i = first; i < numElements; ++i) {
              auto id = digis.moduleInd(i);
              if (id == InvId)
                continue;  // not valid
              if (id != me)
                break;  // end of module
              auto cl = digis.clus(i);
              if (cl < startClus || cl >= lastClus)
                continue;
              cl -= startClus;
              assert(cl >= 0);
              assert(cl < MaxHitsInIter);
              auto x = digis.xx(i);
              auto y = digis.yy(i);
              int32_t ch = std::min(digis.adc(i), pixmx);
              clusterParamsRef.charge[cl] += ch;
              if (clusterParamsRef.minRow[cl] == x)
                clusterParamsRef.Q_f_X[cl] += ch;
              if (clusterParamsRef.maxRow[cl] == x)
                clusterParamsRef.Q_l_X[cl] += ch;
              if (clusterParamsRef.minCol[cl] == y)
                clusterParamsRef.Q_f_Y[cl] += ch;
              if (clusterParamsRef.maxCol[cl] == y)
                clusterParamsRef.Q_l_Y[cl] += ch;
            }

            // next one cluster per thread...

            first = clusters.clusModuleStart(me) + startClus;

            for (int ic = 0; ic < nClusInIter; ++ic) {
              auto h = first + ic;  // output index in global memory

              // this cannot happen anymore
              if (h >= TrackingRecHit2DSOAView::maxHits())
                break;  // overflow...
              assert(h < hits.nHits());
              assert(h < clusters.clusModuleStart(me + 1));

              pixelCPEforGPU::position(cpeParams->commonParams(), cpeParams->detParams(me), clusterParamsRef, ic);
              pixelCPEforGPU::errorFromDB(cpeParams->commonParams(), cpeParams->detParams(me), clusterParamsRef, ic);

              // store it

              hits.charge(h) = clusterParamsRef.charge[ic];

              hits.detectorIndex(h) = me;

              float xl, yl;
              hits.xLocal(h) = xl = clusterParamsRef.xpos[ic];
              hits.yLocal(h) = yl = clusterParamsRef.ypos[ic];

              hits.clusterSizeX(h) = clusterParamsRef.xsize[ic];
              hits.clusterSizeY(h) = clusterParamsRef.ysize[ic];

              hits.xerrLocal(h) = clusterParamsRef.xerr[ic] * clusterParamsRef.xerr[ic];
              hits.yerrLocal(h) = clusterParamsRef.yerr[ic] * clusterParamsRef.yerr[ic];

              // keep it local for computations
              float xg, yg, zg;
              // to global and compute phi...
              cpeParams->detParams(me).frame.toGlobal(xl, yl, xg, yg, zg);
              // here correct for the beamspot...
              xg -= bs->x;
              yg -= bs->y;
              zg -= bs->z;

              hits.xGlobal(h) = xg;
              hits.yGlobal(h) = yg;
              hits.zGlobal(h) = zg;

              hits.rGlobal(h) = std::sqrt(xg * xg + yg * yg);
              hits.iphi(h) = unsafe_atan2s<7>(yg, xg);
            }
          }  // end loop on batches
        });
  }

}  // namespace gpuPixelRecHits

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
