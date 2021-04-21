#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksIterative_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksIterative_h

#include "AlpakaCore/alpakaKernelCommon.h"

#include "AlpakaCore/HistoContainer.h"

#include "gpuVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace gpuVertexFinder {

    // this algo does not really scale as it works in a single block...
    // enough for <10K tracks we have
    struct clusterTracksIterative {
      template <typename T_Acc>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                    ZVertices* pdata,
                                    WorkSpace* pws,
                                    int minT,      // min number of neighbours to be "core"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max  // max normalized distance to cluster
      ) const {
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

        auto& nloops = alpaka::declareSharedVar<int, __COUNTER__>(acc);
        nloops = 0;

        alpaka::syncBlockThreads(acc);

        // cluster seeds only
        bool more = true;
        while (alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, more)) {
          if (1 == nloops % 2) {
            cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
              auto m = iv[i];
              while (m != iv[m])
                m = iv[m];
              iv[i] = m;
            });
          } else {
            more = false;
            cms::alpakatools::for_each_element_in_block_strided(acc, hist.size(), [&](uint32_t k) {
              auto p = hist.begin() + k;
              auto i = (*p);
              auto be = std::min(Hist::bin(izt[i]) + 1, int(hist.nbins() - 1));
              if (nn[i] >= minT) {  // DBSCAN core rule
                auto loop = [&](uint32_t j) {
                  assert(i != j);
                  if (nn[j] < minT)
                    return;  // DBSCAN core rule
                  auto dist = std::abs(zt[i] - zt[j]);
                  if (dist > eps)
                    return;
                  if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
                    return;
                  auto old = alpaka::atomicOp<alpaka::AtomicMin>(acc, &iv[j], iv[i]);
                  if (old != iv[i]) {
                    // end the loop only if no changes were applied
                    more = true;
                  }
                  alpaka::atomicOp<alpaka::AtomicMin>(acc, &iv[i], old);
                };
                ++p;
                for (; p < hist.end(be); ++p)
                  loop(*p);
              }
            });  // for k
          }
          if (threadIdxLocal == 0)
            ++nloops;
        }  // while

        // collect edges (assign to closest cluster of closest point??? here to closest point)
        cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
          //    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
          if (nn[i] < minT) {  // DBSCAN edge rule
            float mdist = eps;
            auto loop = [&](int j) {
              if (nn[j] < minT)
                return;  // DBSCAN core rule
              auto dist = std::abs(zt[i] - zt[j]);
              if (dist > mdist)
                return;
              if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
                return;  // needed?
              mdist = dist;
              iv[i] = iv[j];  // assign to cluster (better be unique??)
            };
            cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
          }
        });

        auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

        foundClusters = 0;
        alpaka::syncBlockThreads(acc);

        // find the number of different clusters, identified by a tracks with clus[i] == i;
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
    };

  }  // namespace gpuVertexFinder

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksIterative_h
