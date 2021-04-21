#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h

#include "AlpakaCore/alpakaKernelCommon.h"

#include "AlpakaCore/HistoContainer.h"

#include "gpuVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace gpuVertexFinder {

    // this algo does not really scale as it works in a single block...
    // enough for <10K tracks we have
    //
    // based on Rodrighez&Laio algo
    //
    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void clusterTracksByDensity(
        const T_Acc& acc,
        gpuVertexFinder::ZVertices* pdata,
        gpuVertexFinder::WorkSpace* pws,
        int minT,      // min number of neighbours to be "seed"
        float eps,     // max absolute distance to cluster
        float errmax,  // max error to be "seed"
        float chi2max  // max normalized distance to cluster
    ) {
      using namespace gpuVertexFinder;
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false

      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      if (verbose && 0 == threadIdxLocal)
        printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);

      auto er2mx = errmax * errmax;

      auto& __restrict__ data = *pdata;
      auto& __restrict__ ws = *pws;
      auto nt = ws.ntrks;
      float const* __restrict__ zt = ws.zt;
      float const* __restrict__ ezt2 = ws.ezt2;

      uint32_t& nvFinal = data.nvFinal;
      uint32_t& nvIntermediate = ws.nvIntermediate;

      uint8_t* __restrict__ izt = ws.izt;
      int32_t* __restrict__ nn = data.ndof;
      int32_t* __restrict__ iv = ws.iv;

      assert(pdata);
      assert(zt);

      using Hist = cms::alpakatools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;
      auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
      auto& hws = alpaka::declareSharedVar<Hist::Counter[32], __COUNTER__>(acc);

      cms::alpakatools::for_each_element_in_block_strided(acc, Hist::totbins(), [&](uint32_t j) { hist.off[j] = 0; });
      alpaka::syncBlockThreads(acc);

      if (verbose && 0 == threadIdxLocal)
        printf("booked hist with %d bins, size %d for %d tracks\n", hist.nbins(), hist.capacity(), nt);

      assert(nt <= hist.capacity());

      // fill hist  (bin shall be wider than "eps")
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        assert(i < ZVertices::MAXTRACKS);
        int iz = int(zt[i] * 10.);  // valid if eps<=0.1
        // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
        iz = std::min(std::max(iz, INT8_MIN), INT8_MAX);
        izt[i] = iz - INT8_MIN;
        assert(iz - INT8_MIN >= 0);
        assert(iz - INT8_MIN < 256);
        hist.count(acc, izt[i]);
        iv[i] = i;
        nn[i] = 0;
      });
      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block(acc, 32, [&](uint32_t i) {
        hws[i] = 0;  // used by prefix scan...
      });

      alpaka::syncBlockThreads(acc);
      hist.finalize(acc, hws);
      alpaka::syncBlockThreads(acc);

      assert(hist.size() == nt);
      cms::alpakatools::for_each_element_in_block_strided(
          acc, nt, [&](uint32_t i) { hist.fill(acc, izt[i], uint16_t(i)); });
      alpaka::syncBlockThreads(acc);

      // count neighbours
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        if (ezt2[i] <= er2mx) {
          auto loop = [&](uint32_t j) {
            if (i == j)
              return;
            auto dist = std::abs(zt[i] - zt[j]);
            if (dist > eps)
              return;
            if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
              return;
            nn[i]++;
          };

          cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
        }
      });

      alpaka::syncBlockThreads(acc);

      // find closest above me .... (we ignore the possibility of two j at same distance from i)
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        float mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < nn[i])
            return;
          if (nn[j] == nn[i] && zt[j] >= zt[i])
            return;  // if equal use natural order...
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;  // (break natural order???)
          mdist = dist;
          iv[i] = j;  // assign to cluster (better be unique??)
        };
        cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
      });

      alpaka::syncBlockThreads(acc);

#ifdef GPU_DEBUG
      //  mini verification
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        if (iv[i] != int(i))
          assert(iv[iv[i]] != int(i));
      });
      alpaka::syncBlockThreads(acc);
#endif

      // consolidate graph (percolate index of seed)
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        auto m = iv[i];
        while (m != iv[m])
          m = iv[m];
        iv[i] = m;
      });

#ifdef GPU_DEBUG
      alpaka::syncBlockThreads(acc);
      //  mini verification
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        if (iv[i] != int(i))
          assert(iv[iv[i]] != int(i));
      });
#endif

#ifdef GPU_DEBUG
      // and verify that we did not spit any cluster...
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        auto minJ = i;
        auto mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < nn[i])
            return;
          if (nn[j] == nn[i] && zt[j] >= zt[i])
            return;  // if equal use natural order...
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;
          mdist = dist;
          minJ = j;
        };
        cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
        // should belong to the same cluster...
        assert(iv[i] == iv[minJ]);
        assert(nn[i] <= nn[iv[i]]);
      });
      alpaka::syncBlockThreads(acc);
#endif

      auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
      foundClusters = 0;
      alpaka::syncBlockThreads(acc);

      // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
      // mark these tracks with a negative id.
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        if (iv[i] == int(i)) {
          if (nn[i] >= minT) {
            auto old = alpaka::atomicOp<alpaka::AtomicInc>(acc, &foundClusters, 0xffffffff);
            iv[i] = -(old + 1);
          } else {  // noise
            iv[i] = -9998;
          }
        }
      });
      alpaka::syncBlockThreads(acc);

      assert(foundClusters < ZVertices::MAXVTX);

      // propagate the negative id to all the tracks in the cluster.
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        if (iv[i] >= 0) {
          // mark each track in a cluster with the same id as the first one
          iv[i] = iv[iv[i]];
        }
      });
      alpaka::syncBlockThreads(acc);

      // adjust the cluster id to be a positive value starting from 0
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) { iv[i] = -iv[i] - 1; });

      nvIntermediate = nvFinal = foundClusters;

      if (verbose && 0 == threadIdxLocal)
        printf("found %d proto vertices\n", foundClusters);
    }

    struct clusterTracksByDensityKernel {
      template <typename T_Acc>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                    gpuVertexFinder::ZVertices* pdata,
                                    gpuVertexFinder::WorkSpace* pws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster
      ) const {
        clusterTracksByDensity(acc, pdata, pws, minT, eps, errmax, chi2max);
      }
    };

  }  // namespace gpuVertexFinder

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h
