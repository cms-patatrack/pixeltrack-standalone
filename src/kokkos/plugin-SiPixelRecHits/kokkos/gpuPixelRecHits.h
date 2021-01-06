#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h

#include <cstdint>
#include <cstdio>
#include <limits>

#include "CondFormats/pixelCPEforGPU.h"
#include "KokkosDataFormats/approx_atan2.h"
#include "KokkosDataFormats/BeamSpotKokkos.h"
#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuPixelRecHits {
    KOKKOS_INLINE_FUNCTION void getHits(pixelCPEforGPU::ParamsOnGPU const* __restrict__ cpeParams,
                                        BeamSpotPOD const* __restrict__ bs,
                                        SiPixelDigisKokkos<KokkosExecSpace>::DeviceConstView pdigis,
                                        int numElements,
                                        SiPixelClustersKokkos<KokkosExecSpace>::DeviceConstView pclusters,
                                        TrackingRecHit2DSOAView* hits,
                                        Kokkos::TeamPolicy<KokkosExecSpace>::member_type const& teamMember) {
      // FIXME
      // the compiler seems NOT to optimize loads from views (even in a simple test case)
      // The whole gimnastic here of copying or not is a pure heuristic exercise that seems to produce the fastest code with the above signature
      // not using views (passing a gazzilion of array pointers) seems to produce the fastest code (but it is harder to mantain)

      assert(hits);

      auto const& digis = pdigis;
      auto const& clusters = pclusters;

      // copy average geometry corrected by beamspot . FIXME (move it somewhere else???)
      if (0 == teamMember.league_rank()) {
        // this is refence in CUDA, but that leads to TeamThreadRange to not compile because the lambda itself is const
        auto agc = &hits->averageGeometry();
        auto const& ag = cpeParams->averageGeometry();
        // workaround for PTHREAD backend
        constexpr auto numberOfLaddersInBarrel = TrackingRecHit2DSOAView::AverageGeometry::numberOfLaddersInBarrel;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, numberOfLaddersInBarrel), [=](int il) {
          agc->ladderZ[il] = ag.ladderZ[il] - bs->z;
          agc->ladderX[il] = ag.ladderX[il] - bs->x;
          agc->ladderY[il] = ag.ladderY[il] - bs->y;
          agc->ladderR[il] = sqrt(agc->ladderX[il] * agc->ladderX[il] + agc->ladderY[il] * agc->ladderY[il]);
          agc->ladderMinZ[il] = ag.ladderMinZ[il] - bs->z;
          agc->ladderMaxZ[il] = ag.ladderMaxZ[il] - bs->z;
        });
        if (0 == teamMember.team_rank()) {
          agc->endCapZ[0] = ag.endCapZ[0] - bs->z;
          agc->endCapZ[1] = ag.endCapZ[1] - bs->z;
          //         printf("endcapZ %f %f\n",agc.endCapZ[0],agc.endCapZ[1]);
        }
      }

      // to be moved in common namespace...
      constexpr uint16_t InvId = 9999;  // must be > MaxNumModules
      constexpr int32_t MaxHitsInIter = pixelCPEforGPU::MaxHitsInIter;

      using ClusParams = pixelCPEforGPU::ClusParams;

      // as usual one block per module
      ClusParams* clusParams = reinterpret_cast<ClusParams*>(teamMember.team_shmem().get_shmem(sizeof(ClusParams)));

      auto me = clusters.moduleId(teamMember.league_rank());
      int nclus = clusters.clusInModule(me);

      if (0 == nclus)
        return;

#ifdef GPU_DEBUG
      if (teamMember.team_rank() == 0) {
        auto k = clusters.moduleStart(1 + teamMember.league_rank());
        while (digis.moduleInd(k) == InvId)
          ++k;
        assert(digis.moduleInd(k) == me);
      }
#endif

#ifdef GPU_DEBUG
      if (me % 100 == 1)
        if (teamMember.team_rank() == 0)
          printf("hitbuilder: %d clusters in module %d. will write at %d\n", nclus, me, clusters.clusModuleStart(me));
#endif

      for (int startClus = 0, endClus = nclus; startClus < endClus; startClus += MaxHitsInIter) {
        auto first = clusters.moduleStart(1 + teamMember.league_rank());

        int nClusInIter = std::min(MaxHitsInIter, endClus - startClus);
        int lastClus = startClus + nClusInIter;
        assert(nClusInIter <= nclus);
        assert(nClusInIter > 0);
        assert(lastClus <= nclus);

        assert(nclus > MaxHitsInIter || (0 == startClus && nClusInIter == nclus && lastClus == nclus));

        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, nClusInIter), [=](int ic) {
          clusParams->minRow[ic] = std::numeric_limits<uint32_t>::max();
          clusParams->maxRow[ic] = 0;
          clusParams->minCol[ic] = std::numeric_limits<uint32_t>::max();
          clusParams->maxCol[ic] = 0;
          clusParams->charge[ic] = 0;
          clusParams->Q_f_X[ic] = 0;
          clusParams->Q_l_X[ic] = 0;
          clusParams->Q_f_Y[ic] = 0;
          clusParams->Q_l_Y[ic] = 0;
        });

        first += teamMember.team_rank();
        teamMember.team_barrier();

        // one thead per "digi"

        // TODO: can't use parallel_for+TeamThreadRange because of the break
        for (int i = first; i < numElements; i += teamMember.team_size()) {
          auto id = digis.moduleInd(i);
          if (id == InvId)
            continue;
          if (id != me)
            break;  // end of module
          auto cl = digis.clus(i);
          if (cl < startClus || cl >= lastClus)
            continue;
          auto x = digis.xx(i);
          auto y = digis.yy(i);
          cl -= startClus;
          assert(cl >= 0);
          assert(cl < MaxHitsInIter);
          Kokkos::atomic_min_fetch<uint32_t>(&clusParams->minRow[cl], x);
          Kokkos::atomic_max_fetch<uint32_t>(&clusParams->maxRow[cl], x);
          Kokkos::atomic_min_fetch<uint32_t>(&clusParams->minCol[cl], y);
          Kokkos::atomic_max_fetch<uint32_t>(&clusParams->maxCol[cl], y);
        }

        teamMember.team_barrier();

        // pixmx is not available in the binary dumps
        //auto pixmx = cpeParams->detParams(me).pixmx;
        auto pixmx = std::numeric_limits<uint16_t>::max();
        // TODO: can't use parallel_for+TeamThreadRange because of the break
        for (int i = first; i < numElements; i += teamMember.team_size()) {
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
          auto ch = std::min(digis.adc(i), pixmx);
          Kokkos::atomic_add<int32_t>(&clusParams->charge[cl], ch);
          if (clusParams->minRow[cl] == x)
            Kokkos::atomic_add<int32_t>(&clusParams->Q_f_X[cl], ch);
          if (clusParams->maxRow[cl] == x)
            Kokkos::atomic_add<int32_t>(&clusParams->Q_l_X[cl], ch);
          if (clusParams->minCol[cl] == y)
            Kokkos::atomic_add<int32_t>(&clusParams->Q_f_Y[cl], ch);
          if (clusParams->maxCol[cl] == y)
            Kokkos::atomic_add<int32_t>(&clusParams->Q_l_Y[cl], ch);
        }

        teamMember.team_barrier();
        // next one cluster per thread...

        first = clusters.clusModuleStart(me) + startClus;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, nClusInIter), [=](int ic) {
          auto h = first + ic;  // output index in global memory

          // this cannot happen anymore
          if (h >= TrackingRecHit2DSOAView::maxHits())
            // TODO: was 'break', OTOH comment above says "should not happen", so maybe 'return' is ok
            return;  // overflow...
          assert(h < hits->nHits());
          assert(h < clusters.clusModuleStart(me + 1));

          pixelCPEforGPU::position(cpeParams->commonParams(), cpeParams->detParams(me), *clusParams, ic);
          pixelCPEforGPU::errorFromDB(cpeParams->commonParams(), cpeParams->detParams(me), *clusParams, ic);

          // store it

          hits->charge(h) = clusParams->charge[ic];

          hits->detectorIndex(h) = me;

          float xl, yl;
          hits->xLocal(h) = xl = clusParams->xpos[ic];
          hits->yLocal(h) = yl = clusParams->ypos[ic];

          hits->clusterSizeX(h) = clusParams->xsize[ic];
          hits->clusterSizeY(h) = clusParams->ysize[ic];

          hits->xerrLocal(h) = clusParams->xerr[ic] * clusParams->xerr[ic];
          hits->yerrLocal(h) = clusParams->yerr[ic] * clusParams->yerr[ic];

          // keep it local for computations
          float xg, yg, zg;
          // to global and compute phi...
          cpeParams->detParams(me).frame.toGlobal(xl, yl, xg, yg, zg);
          // here correct for the beamspot...
          xg -= bs->x;
          yg -= bs->y;
          zg -= bs->z;

          hits->xGlobal(h) = xg;
          hits->yGlobal(h) = yg;
          hits->zGlobal(h) = zg;

          hits->rGlobal(h) = std::sqrt(xg * xg + yg * yg);
          hits->iphi(h) = unsafe_atan2s<7>(yg, xg);
        });
        teamMember.team_barrier();
      }  // end loop on batches
    }
  }  // namespace gpuPixelRecHits
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h
