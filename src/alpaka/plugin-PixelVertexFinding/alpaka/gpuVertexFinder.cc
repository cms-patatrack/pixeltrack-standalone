//#include <iostream>

#include "AlpakaCore/config.h"
#include "AlpakaCore/memory.h"
#include "AlpakaCore/workdivision.h"

#include "gpuVertexFinder.h"
#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace gpuVertexFinder {

    struct loadTracks {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(
          const TAcc& acc, TkSoA const* ptracks, ZVertexSoA* soa, WorkSpace* pws, float ptMin) const {
        ALPAKA_ASSERT_ACC(ptracks);
        ALPAKA_ASSERT_ACC(soa);
        auto const& tracks = *ptracks;
        auto const& fit = tracks.stateAtBS;
        auto const* quality = tracks.qualityData();

        cms::alpakatools::for_each_element_in_grid_strided(acc, TkSoA::stride(), [&](uint32_t idx) {
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
          auto it = alpaka::atomicAdd(acc, &data.ntrks, 1u, alpaka::hierarchy::Blocks{});
          data.itrk[it] = idx;
          data.zt[it] = tracks.zip(idx);
          data.ezt2[it] = fit.covariance(idx)(14);
          data.ptt2[it] = pt * pt;
        });
      }
    };

// #define THREE_KERNELS
#ifndef THREE_KERNELS
    struct vertexFinderOneKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    gpuVertexFinder::ZVertices* pdata,
                                    gpuVertexFinder::WorkSpace* pws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster
      ) const {
        clusterTracksByDensity(acc, pdata, pws, minT, eps, errmax, chi2max);
        alpaka::syncBlockThreads(acc);
        fitVertices(acc, pdata, pws, 50.);
        alpaka::syncBlockThreads(acc);
        splitVertices(acc, pdata, pws, 9.f);
        alpaka::syncBlockThreads(acc);
        fitVertices(acc, pdata, pws, 5000.);
        alpaka::syncBlockThreads(acc);
        sortByPt2(acc, pdata, pws);
      }
    };
#else
    struct vertexFinderKernel1 {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    gpuVertexFinder::ZVertices* pdata,
                                    gpuVertexFinder::WorkSpace* pws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster,
      ) const {
        clusterTracksByDensity(acc, pdata, pws, minT, eps, errmax, chi2max);
        alpaka::syncBlockThreads(acc);
        fitVertices(acc, pdata, pws, 50.);
      }
    };

    struct vertexFinderKernel2 {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    gpuVertexFinder::ZVertices* pdata,
                                    gpuVertexFinder::WorkSpace* pws) const {
        fitVertices(acc, pdata, pws, 5000.);
        alpaka::syncBlockThreads(acc);
        sortByPt2(acc, pdata, pws);
      }
    };
#endif

    ZVertexAlpaka Producer::makeAsync(TkSoA const* tksoa, float ptMin, Queue& queue) const {
      // std::cout << "producing Vertices on GPU" << std::endl;
      ALPAKA_ASSERT_ACC(tksoa);

      ZVertexAlpaka vertices = cms::alpakatools::make_device_buffer<ZVertexSoA>(queue);
      auto* soa = vertices.data();
      ALPAKA_ASSERT_ACC(soa);

      auto ws_dBuf{cms::alpakatools::make_device_buffer<WorkSpace>(queue)};
      auto ws_d = ws_dBuf.data();

      auto nvFinalVerticesView = cms::alpakatools::make_device_view(alpaka::getDev(queue), soa->nvFinal);
      alpaka::memset(queue, nvFinalVerticesView, 0);
      auto ntrksWorkspaceView = cms::alpakatools::make_device_view(alpaka::getDev(queue), ws_d->ntrks);
      alpaka::memset(queue, ntrksWorkspaceView, 0);
      auto nvIntermediateWorkspaceView =
          cms::alpakatools::make_device_view(alpaka::getDev(queue), ws_d->nvIntermediate);
      alpaka::memset(queue, nvIntermediateWorkspaceView, 0);

      const uint32_t blockSize = 128;
      const uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(TkSoA::stride(), blockSize);
      const auto loadTracksWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(loadTracksWorkDiv, loadTracks(), tksoa, soa, ws_d, ptMin));

      const auto finderSorterWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024 - 256);
      const auto splitterFitterWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1024, 128);

      if (oneKernel_) {
        // implemented only for density clustesrs
#ifndef THREE_KERNELS
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(
                            finderSorterWorkDiv, vertexFinderOneKernel(), soa, ws_d, minT, eps, errmax, chi2max));

#else
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(
                            finderSorterWorkDiv, vertexFinderKernel1(), soa, ws_d, minT, eps, errmax, chi2max));
        // one block per vertex...
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(splitterFitterWorkDiv, splitVerticesKernel(), soa, ws_d, 9.f));

        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(finderSorterWorkDiv, vertexFinderKernel2(), soa, ws_d));
#endif

      } else {  // five kernels

        if (useDensity_) {
          alpaka::enqueue(
              queue,
              alpaka::createTaskKernel<Acc1D>(
                  finderSorterWorkDiv, clusterTracksByDensityKernel(), soa, ws_d, minT, eps, errmax, chi2max));
        } else if (useDBSCAN_) {
          alpaka::enqueue(queue,
                          alpaka::createTaskKernel<Acc1D>(
                              finderSorterWorkDiv, clusterTracksDBSCAN(), soa, ws_d, minT, eps, errmax, chi2max));
        } else if (useIterative_) {
          alpaka::enqueue(queue,
                          alpaka::createTaskKernel<Acc1D>(
                              finderSorterWorkDiv, clusterTracksIterative(), soa, ws_d, minT, eps, errmax, chi2max));
        }

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(finderSorterWorkDiv, fitVerticesKernel(), soa, ws_d, 50.));
        // one block per vertex...
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(splitterFitterWorkDiv, splitVerticesKernel(), soa, ws_d, 9.f));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(finderSorterWorkDiv, fitVerticesKernel(), soa, ws_d, 5000.));

        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(finderSorterWorkDiv, sortByPt2Kernel(), soa, ws_d));
      }

      return vertices;
    }

  }  // namespace gpuVertexFinder

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
