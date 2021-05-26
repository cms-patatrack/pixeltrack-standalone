//
// Author: Felice Pantaleo, CERN
//

// #define BROKENLINE_DEBUG

#include <cstdint>

#include "AlpakaCore/alpakaKernelCommon.h"

#include "AlpakaDataFormats/TrackingRecHit2DAlpaka.h"
#include "CondFormats/pixelCPEforGPU.h"

#include "BrokenLine.h"
#include "HelixFitOnGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using HitsOnGPU = TrackingRecHit2DSOAView;
  using Tuples = pixelTrack::HitContainer;
  using OutputSoA = pixelTrack::TrackSoA;

  // #define BL_DUMP_HITS

  template <int N>
  struct kernelBLFastFit {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  Tuples const *__restrict__ foundNtuplets,
                                  CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                  HitsOnGPU const *__restrict__ hhp,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit,
                                  uint32_t nHits,
                                  uint32_t offset) const {
      constexpr uint32_t hitsInFit = N;

      assert(hitsInFit <= nHits);

      assert(hhp);
      assert(pfast_fit);
      assert(foundNtuplets);
      assert(tupleMultiplicity);

      // look in bin for this hit multiplicity
#ifdef BROKENLINE_DEBUG
      const uint32_t threadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
      if (0 == threadIdx) {
        printf("%d total Ntuple\n", foundNtuplets->nbins());
        printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
      }
#endif

      const auto nt = Rfit::maxNumberOfConcurrentFits();
      cms::alpakatools::for_each_element_in_grid_strided(acc, nt, [&](uint32_t local_idx) {
        auto tuple_idx = local_idx + offset;
        if (tuple_idx >= tupleMultiplicity->size(nHits))
          return;

        // get it from the ntuple container (one to one to helix)
        auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
        assert(tkid < foundNtuplets->nbins());

        assert(foundNtuplets->size(tkid) == nHits);

        Rfit::Map3xNd<N> hits(phits + local_idx);
        Rfit::Map4d fast_fit(pfast_fit + local_idx);
        Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);

#ifdef BL_DUMP_HITS
        auto &&done = alpaka::declareSharedVar<int, __COUNTER__>(acc);
        done = 0;
        alpaka::syncBlockThreads(acc);
        bool dump = (foundNtuplets->size(tkid) == 5 && 0 == alpaka::atomicOp<alpaka::AtomicAdd>(acc, &done, 1));
#endif

        // Prepare data structure
        auto const *hitId = foundNtuplets->begin(tkid);
        for (unsigned int i = 0; i < hitsInFit; ++i) {
          auto hit = hitId[i];
          float ge[6];
          hhp->cpeParams()
              .detParams(hhp->detectorIndex(hit))
              .frame.toGlobal(hhp->xerrLocal(hit), 0, hhp->yerrLocal(hit), ge);
#ifdef BL_DUMP_HITS
          if (dump) {
            printf("Hit global: %d: %d hits.col(%d) << %f,%f,%f\n",
                   tkid,
                   hhp->detectorIndex(hit),
                   i,
                   hhp->xGlobal(hit),
                   hhp->yGlobal(hit),
                   hhp->zGlobal(hit));
            printf("Error: %d: %d  hits_ge.col(%d) << %e,%e,%e,%e,%e,%e\n",
                   tkid,
                   hhp->detetectorIndex(hit),
                   i,
                   ge[0],
                   ge[1],
                   ge[2],
                   ge[3],
                   ge[4],
                   ge[5]);
          }
#endif
          hits.col(i) << hhp->xGlobal(hit), hhp->yGlobal(hit), hhp->zGlobal(hit);
          hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
        }
        BrokenLine::BL_Fast_fit(hits, fast_fit);

        // no NaN here....
        assert(fast_fit(0) == fast_fit(0));
        assert(fast_fit(1) == fast_fit(1));
        assert(fast_fit(2) == fast_fit(2));
        assert(fast_fit(3) == fast_fit(3));
      });

    }  // kernel operator()
  };   // struct

  template <int N>
  struct kernelBLFit {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                  double B,
                                  OutputSoA *results,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit,
                                  uint32_t nHits,
                                  uint32_t offset) const {
      assert(N <= nHits);

      assert(results);
      assert(pfast_fit);

      // same as above...

      // look in bin for this hit multiplicity
      //auto local_start = blockIdx.x * blockDim.x + threadIdx.x;
      //for (int local_idx = local_start, nt = Rfit::maxNumberOfConcurrentFits(); local_idx < nt;
      //local_idx += gridDim.x * blockDim.x) {
      const auto nt = Rfit::maxNumberOfConcurrentFits();
      cms::alpakatools::for_each_element_in_grid_strided(acc, nt, [&](uint32_t local_idx) {
        auto tuple_idx = local_idx + offset;
        if (tuple_idx >= tupleMultiplicity->size(nHits))
          return;

        // get it for the ntuple container (one to one to helix)
        auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);

        Rfit::Map3xNd<N> hits(phits + local_idx);
        Rfit::Map4d fast_fit(pfast_fit + local_idx);
        Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        BrokenLine::PreparedBrokenLineData<N> data;
        Rfit::Matrix3d Jacob;

        BrokenLine::karimaki_circle_fit circle;
        Rfit::line_fit line;

        BrokenLine::prepareBrokenLineData(hits, fast_fit, B, data);
        BrokenLine::BL_Line_fit(hits_ge, fast_fit, B, data, line);
        BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit, B, data, circle);

        results->stateAtBS.copyFromCircle(circle.par, circle.cov, line.par, line.cov, 1.f / float(B), tkid);
        results->pt(tkid) = float(B) / float(std::abs(circle.par(2)));
        results->eta(tkid) = asinhf(line.par(0));
        results->chi2(tkid) = (circle.chi2 + line.chi2) / (2 * N - 5);

#ifdef BROKENLINE_DEBUG
        if (!(circle.chi2 >= 0) || !(line.chi2 >= 0))
          printf("kernelBLFit failed! %f/%f\n", circle.chi2, line.chi2);
        printf("kernelBLFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
               N,
               nHits,
               tkid,
               circle.par(0),
               circle.par(1),
               circle.par(2));
        printf("kernelBLHits line.par(0,1): %d %f,%f\n", tkid, line.par(0), line.par(1));
        printf("kernelBLHits chi2 cov %f/%f  %e,%e,%e,%e,%e\n",
               circle.chi2,
               line.chi2,
               circle.cov(0, 0),
               circle.cov(1, 1),
               circle.cov(2, 2),
               line.cov(0, 0),
               line.cov(1, 1));
#endif
      });

    }  // kernel operator()
  };   // struct

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
