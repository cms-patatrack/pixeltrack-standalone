#ifndef plugin_SiPixelRecHits_alpaka_gpuPixelRecHits_h
#define plugin_SiPixelRecHits_alpaka_gpuPixelRecHits_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "AlpakaCore/config.h"
#include "AlpakaDataFormats/alpaka/BeamSpotAlpaka.h"
#include "AlpakaDataFormats/alpaka/TrackingRecHit2DAlpaka.h"
#include "CondFormats/pixelCPEforGPU.h"
#include "DataFormats/approx_atan2.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace gpuPixelRecHits {

    struct getHits {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    pixelCPEforGPU::ParamsOnGPU const* __restrict__ cpeParams,
                                    BeamSpotPOD const* __restrict__ bs,
                                    SiPixelDigisAlpaka::DeviceConstView const digis,
                                    uint32_t numElements,
                                    SiPixelClustersAlpaka::DeviceConstView const clusters,
                                    TrackingRecHit2DSoAView* phits) const {
        // FIXME
        // the compiler seems NOT to optimize loads from views (even in a simple test case)
        // The whole gimnastic here of copying or not is a pure heuristic exercise that seems to produce the fastest code with the above signature
        // not using views (passing a gazzilion of array pointers) seems to produce the fastest code (but it is harder to mantain)

        ALPAKA_ASSERT_ACC(phits);
        ALPAKA_ASSERT_ACC(cpeParams);

        auto& hits = *phits;

        const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // copy average geometry corrected by beamspot . FIXME (move it somewhere else???)
        if (0 == blockIdx) {
          auto& agc = hits.averageGeometry();
          auto const& ag = cpeParams->averageGeometry();
          constexpr auto numberOfLaddersInBarrel = TrackingRecHit2DSoAView::AverageGeometry::numberOfLaddersInBarrel;
          cms::alpakatools::for_each_element_in_block_strided(acc, numberOfLaddersInBarrel, [&](uint32_t il) {
            agc.ladderX[il] = ag.ladderX[il] - bs->x;
            agc.ladderY[il] = ag.ladderY[il] - bs->y;
            agc.ladderZ[il] = ag.ladderZ[il] - bs->z;
            agc.ladderR[il] = sqrt(agc.ladderX[il] * agc.ladderX[il] + agc.ladderY[il] * agc.ladderY[il]);
            agc.ladderMinZ[il] = ag.ladderMinZ[il] - bs->z;
            agc.ladderMaxZ[il] = ag.ladderMaxZ[il] - bs->z;
          });
          if (threadIdxLocal == 0) {
            agc.endCapZ[0] = ag.endCapZ[0] - bs->z;
            agc.endCapZ[1] = ag.endCapZ[1] - bs->z;
            //printf("endcapZ %f %f\n",agc.endCapZ[0],agc.endCapZ[1]);
          }
        }

        // to be moved in common namespace...
        constexpr uint16_t InvId = 9999;  // must be > MaxNumModules
        constexpr int32_t MaxHitsInIter = pixelCPEforGPU::MaxHitsInIter;

        using ClusParams = pixelCPEforGPU::ClusParams;

        // as usual one block per module
        auto& clusParams = alpaka::declareSharedVar<ClusParams, __COUNTER__>(acc);

        auto me = clusters.moduleId(blockIdx);
        int nclus = clusters.clusInModule(me);

        if (0 == nclus)
          return;

#ifdef GPU_DEBUG
        if (threadIdxLocal == 0) {
          auto k = clusters.moduleStart(1 + blockIdx);
          while (digis.moduleInd(k) == InvId)
            ++k;
          ALPAKA_ASSERT_ACC(digis.moduleInd(k) == me);
        }
#endif

#ifdef GPU_DEBUG
        if (me % 100 == 1)
          if (threadIdxLocal == 0)
            printf("hitbuilder: %d clusters in module %d. will write at %d\n", nclus, me, clusters.clusModuleStart(me));
#endif

        for (int startClus = 0, endClus = nclus; startClus < endClus; startClus += MaxHitsInIter) {
          auto first = clusters.moduleStart(1 + blockIdx);

          int nClusInIter = std::min(MaxHitsInIter, endClus - startClus);
          int lastClus = startClus + nClusInIter;
          ALPAKA_ASSERT_ACC(nClusInIter <= nclus);
          ALPAKA_ASSERT_ACC(nClusInIter > 0);
          ALPAKA_ASSERT_ACC(lastClus <= nclus);
          ALPAKA_ASSERT_ACC(nclus > MaxHitsInIter || (0 == startClus && nClusInIter == nclus && lastClus == nclus));

          // init
          cms::alpakatools::for_each_element_in_block_strided(acc, nClusInIter, [&](uint32_t ic) {
            clusParams.minRow[ic] = std::numeric_limits<uint32_t>::max();
            clusParams.maxRow[ic] = 0;
            clusParams.minCol[ic] = std::numeric_limits<uint32_t>::max();
            clusParams.maxCol[ic] = 0;
            clusParams.charge[ic] = 0;
            clusParams.Q_f_X[ic] = 0;
            clusParams.Q_l_X[ic] = 0;
            clusParams.Q_f_Y[ic] = 0;
            clusParams.Q_l_Y[ic] = 0;
          });

          alpaka::syncBlockThreads(acc);

          // one thread per "digi"
          const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
          const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
              cms::alpakatools::element_index_range_in_block(acc, first);
          uint32_t rowsColsFirstElementIdx = firstElementIdxNoStride;
          uint32_t rowsColsEndElementIdx = endElementIdxNoStride;
          for (uint32_t i = rowsColsFirstElementIdx; i < numElements; ++i) {
            if (not cms::alpakatools::next_valid_element_index_strided(
                    i, rowsColsFirstElementIdx, rowsColsEndElementIdx, blockDimension, numElements))
              break;
            auto id = digis.moduleInd(i);
            if (id == InvId)
              continue;  // not valid
            if (id != me)
              break;  // end of module
            auto cl = digis.clus(i);
            if (cl < startClus || cl >= lastClus)
              continue;
            const uint32_t x = digis.xx(i);
            const uint32_t y = digis.yy(i);
            cl -= startClus;
            ALPAKA_ASSERT_ACC(cl >= 0);
            ALPAKA_ASSERT_ACC(cl < MaxHitsInIter);
            alpaka::atomicMin(acc, &clusParams.minRow[cl], x, alpaka::hierarchy::Threads{});
            alpaka::atomicMax(acc, &clusParams.maxRow[cl], x, alpaka::hierarchy::Threads{});
            alpaka::atomicMin(acc, &clusParams.minCol[cl], y, alpaka::hierarchy::Threads{});
            alpaka::atomicMax(acc, &clusParams.maxCol[cl], y, alpaka::hierarchy::Threads{});
          }

          alpaka::syncBlockThreads(acc);

          // pixmx is not available in the binary dumps
          //auto pixmx = cpeParams->detParams(me).pixmx;
          auto pixmx = std::numeric_limits<uint16_t>::max();
          uint32_t chargeFirstElementIdx = firstElementIdxNoStride;
          uint32_t chargeEndElementIdx = endElementIdxNoStride;
          for (uint32_t i = chargeFirstElementIdx; i < numElements; ++i) {
            if (not cms::alpakatools::next_valid_element_index_strided(
                    i, chargeFirstElementIdx, chargeEndElementIdx, blockDimension, numElements))
              break;
            auto id = digis.moduleInd(i);
            if (id == InvId)
              continue;  // not valid
            if (id != me)
              break;  // end of module
            auto cl = digis.clus(i);
            if (cl < startClus || cl >= lastClus)
              continue;
            cl -= startClus;
            ALPAKA_ASSERT_ACC(cl >= 0);
            ALPAKA_ASSERT_ACC(cl < MaxHitsInIter);
            const uint32_t x = digis.xx(i);
            const uint32_t y = digis.yy(i);
            const int32_t ch = std::min(digis.adc(i), pixmx);
            alpaka::atomicAdd(acc, &clusParams.charge[cl], ch, alpaka::hierarchy::Threads{});
            if (clusParams.minRow[cl] == x)
              alpaka::atomicAdd(acc, &clusParams.Q_f_X[cl], ch, alpaka::hierarchy::Threads{});
            if (clusParams.maxRow[cl] == x)
              alpaka::atomicAdd(acc, &clusParams.Q_l_X[cl], ch, alpaka::hierarchy::Threads{});
            if (clusParams.minCol[cl] == y)
              alpaka::atomicAdd(acc, &clusParams.Q_f_Y[cl], ch, alpaka::hierarchy::Threads{});
            if (clusParams.maxCol[cl] == y)
              alpaka::atomicAdd(acc, &clusParams.Q_l_Y[cl], ch, alpaka::hierarchy::Threads{});
          }

          alpaka::syncBlockThreads(acc);

          // next one cluster per thread...
          first = clusters.clusModuleStart(me) + startClus;

          cms::alpakatools::for_each_element_in_block_strided(acc, nClusInIter, [&](uint32_t ic) {
            auto h = first + ic;  // output index in global memory

            // this cannot happen anymore
            // TODO: was 'break', OTOH comment above says "should not happen", so hopefully 'return' is ok
            if (h >= TrackingRecHit2DSoAView::maxHits()) {
              return;  // overflow...
            }
            ALPAKA_ASSERT_ACC(h < hits.nHits());
            ALPAKA_ASSERT_ACC(h < clusters.clusModuleStart(me + 1));

            pixelCPEforGPU::position(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);
            pixelCPEforGPU::errorFromDB(cpeParams->commonParams(), cpeParams->detParams(me), clusParams, ic);

            // store it
            hits.charge(h) = clusParams.charge[ic];
            hits.detectorIndex(h) = me;
            float xl, yl;
            hits.xLocal(h) = xl = clusParams.xpos[ic];
            hits.yLocal(h) = yl = clusParams.ypos[ic];
            hits.clusterSizeX(h) = clusParams.xsize[ic];
            hits.clusterSizeY(h) = clusParams.ysize[ic];
            hits.xerrLocal(h) = clusParams.xerr[ic] * clusParams.xerr[ic];
            hits.yerrLocal(h) = clusParams.yerr[ic] * clusParams.yerr[ic];

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
          });

          alpaka::syncBlockThreads(acc);
        }  // end loop on batches
      }    // end of kernel operator()
    };

  }  // namespace gpuPixelRecHits
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_SiPixelRecHits_alpaka_gpuPixelRecHits_h
