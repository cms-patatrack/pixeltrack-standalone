#ifndef plugin_SiPixelClusterizer_alpaka_gpuClustering_h
#define plugin_SiPixelClusterizer_alpaka_gpuClustering_h

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "AlpakaCore/config.h"
#include "AlpakaCore/HistoContainer.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"
#include "Geometry/phase1PixelTopology.h"

namespace gpuClustering {

#ifdef GPU_DEBUG

#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)
#define GLOBAL_STATIC_VAR __device__ static
#else
#define GLOBAL_STATIC_VAR static
#endif

  template <typename TAcc>
  GLOBAL_STATIC_VAR uint32_t gMaxHit = 0;
#endif

  struct countModules {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  uint16_t const* __restrict__ id,
                                  uint32_t* __restrict__ moduleStart,
                                  int32_t* __restrict__ clusterId,
                                  const unsigned int numElements) const {
      cms::alpakatools::for_each_element_in_grid_strided(acc, numElements, [&](uint32_t i) {
        clusterId[i] = i;
        if (InvId != id[i]) {
          int j = i - 1;
          while (j >= 0 and id[j] == InvId)
            --j;
          if (j < 0 or id[j] != id[i]) {
            // boundary...
            auto loc =
                alpaka::atomicInc(acc, moduleStart, std::decay_t<uint32_t>(MaxNumModules), alpaka::hierarchy::Blocks{});

            moduleStart[loc + 1] = i;
          }
        }
      });
    }
  };

  //  __launch_bounds__(256,4)
  struct findClus {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc,
        uint16_t const* __restrict__ id,           // module id of each pixel
        uint16_t const* __restrict__ x,            // local coordinates of each pixel
        uint16_t const* __restrict__ y,            //
        uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
        uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
        uint32_t* __restrict__ moduleId,           // output: module id of each module
        int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
        const unsigned int numElements) const {
      const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      if (blockIdx >= moduleStart[0])
        return;

      auto firstPixel = moduleStart[1 + blockIdx];
      auto thisModuleId = id[firstPixel];
      ALPAKA_ASSERT_ACC(thisModuleId < MaxNumModules);

      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
        if (threadIdxLocal == 0)
          printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx);
#endif

      // find the index of the first pixel not belonging to this module (or invalid)
      auto& msize = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
      msize = numElements;
      alpaka::syncBlockThreads(acc);

      // Stride = block size.
      const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

      // Get thread / CPU element indices in block.
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
          cms::alpakatools::element_index_range_in_block(acc, firstPixel);
      uint32_t firstElementIdx = firstElementIdxNoStride;
      uint32_t endElementIdx = endElementIdxNoStride;

      // skip threads not associated to an existing pixel
      for (uint32_t i = firstElementIdx; i < numElements; ++i) {
        if (not cms::alpakatools::next_valid_element_index_strided(
                i, firstElementIdx, endElementIdx, blockDimension, numElements))
          break;
        if (id[i] == InvId)  // skip invalid pixels
          continue;
        if (id[i] != thisModuleId) {  // find the first pixel in a different module
          alpaka::atomicMin(acc, &msize, i, alpaka::hierarchy::Threads{});
          break;
        }
      }

      //init hist  (ymax=416 < 512 : 9bits)
      constexpr uint32_t maxPixInModule = 4000;
      constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
      using Hist = cms::alpakatools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
      auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
      auto& ws = alpaka::declareSharedVar<Hist::Counter[32], __COUNTER__>(acc);

      cms::alpakatools::for_each_element_in_block_strided(acc, Hist::totbins(), [&](uint32_t j) { hist.off[j] = 0; });
      alpaka::syncBlockThreads(acc);

      ALPAKA_ASSERT_ACC((msize == numElements) or ((msize < numElements) and (id[msize] != thisModuleId)));

      // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
      if (0 == threadIdxLocal) {
        if (msize - firstPixel > maxPixInModule) {
          printf("too many pixels in module %d: %d > %d\n", thisModuleId, msize - firstPixel, maxPixInModule);
          msize = maxPixInModule + firstPixel;
        }
      }

      alpaka::syncBlockThreads(acc);
      ALPAKA_ASSERT_ACC(msize - firstPixel <= maxPixInModule);

#ifdef GPU_DEBUG
      auto& totGood = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      totGood = 0;
      alpaka::syncBlockThreads(acc);
#endif

      // fill histo
      cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
        if (id[i] != InvId) {  // skip invalid pixels
          hist.count(acc, y[i]);
#ifdef GPU_DEBUG
          alpaka::atomicAdd(acc, &totGood, 1u, alpaka::hierarchy::Blocks{});
#endif
        }
      });
      alpaka::syncBlockThreads(acc);
      cms::alpakatools::for_each_element_in_block(acc, 32u, [&](uint32_t i) {
        ws[i] = 0;  // used by prefix scan...
      });
      alpaka::syncBlockThreads(acc);
      hist.finalize(acc, ws);
      alpaka::syncBlockThreads(acc);
#ifdef GPU_DEBUG
      ALPAKA_ASSERT_ACC(hist.size() == totGood);
      if (thisModuleId % 100 == 1)
        if (threadIdxLocal == 0)
          printf("histo size %d\n", hist.size());
#endif
      cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
        if (id[i] != InvId) {  // skip invalid pixels
          hist.fill(acc, y[i], i - firstPixel);
        }
      });

      // Assume that we can cover the whole module with up to 16 blockDimension-wide iterations
      // This maxiter value was tuned for GPU, with 256 or 512 threads per block.
      // Hence, also works for CPU case, with 256 or 512 elements per thread.
      // Real constrainst is maxiter = hist.size() / blockDimension,
      // with blockDimension = threadPerBlock * elementsPerThread.
      // Hence, maxiter can be tuned accordingly to the workdiv.
      constexpr unsigned int maxiter = 16;
      ALPAKA_ASSERT_ACC((hist.size() / blockDimension) <= maxiter);

#if defined(ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND) || defined(ALPAKA_ACC_GPU_HIP_ASYNC_BACKEND)
      constexpr uint32_t threadDimension = 1;
#else
      // NB: can be tuned.
      constexpr uint32_t threadDimension = 256;
#endif

#ifndef NDEBUG
      [[maybe_unused]] const uint32_t runTimeThreadDimension(
          alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
      ALPAKA_ASSERT_ACC(runTimeThreadDimension <= threadDimension);
#endif

      // nearest neighbour
      // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
      constexpr int maxNeighbours = 10;
      uint16_t nn[maxiter][threadDimension][maxNeighbours];
      uint8_t nnn[maxiter][threadDimension];  // number of nn
      for (uint32_t elementIdx = 0; elementIdx < threadDimension; ++elementIdx) {
        for (uint32_t k = 0; k < maxiter; ++k) {
          nnn[k][elementIdx] = 0;
        }
      }

      alpaka::syncBlockThreads(acc);  // for hit filling!

#ifdef GPU_DEBUG
      // look for anomalous high occupancy
      auto& n40 = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      auto& n60 = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      n40 = n60 = 0;
      alpaka::syncBlockThreads(acc);
      cms::alpakatools::for_each_element_in_block_strided(acc, Hist::nbins(), [&](uint32_t j) {
        if (hist.size(j) > 60)
          alpaka::atomicAdd(acc, &n60, 1u, alpaka::hierarchy::Blocks{});
        if (hist.size(j) > 40)
          alpaka::atomicAdd(acc, &n40, 1u, alpaka::hierarchy::Blocks{});
      });
      alpaka::syncBlockThreads(acc);
      if (0 == threadIdxLocal) {
        if (n60 > 0)
          printf("columns with more than 60 px %d in %d\n", n60, thisModuleId);
        else if (n40 > 0)
          printf("columns with more than 40 px %d in %d\n", n40, thisModuleId);
      }
      alpaka::syncBlockThreads(acc);
#endif

      // fill NN
      uint32_t k = 0u;
      cms::alpakatools::for_each_element_in_block_strided(acc, hist.size(), [&](uint32_t j) {
        const uint32_t jEquivalentClass = j % threadDimension;
        k = j / blockDimension;
        ALPAKA_ASSERT_ACC(k < maxiter);
        auto p = hist.begin() + j;
        auto i = *p + firstPixel;
        ALPAKA_ASSERT_ACC(id[i] != InvId);
        ALPAKA_ASSERT_ACC(id[i] == thisModuleId);  // same module
        int be = Hist::bin(y[i] + 1);
        auto e = hist.end(be);
        ++p;
        ALPAKA_ASSERT_ACC(0 == nnn[k][jEquivalentClass]);
        for (; p < e; ++p) {
          auto m = (*p) + firstPixel;
          ALPAKA_ASSERT_ACC(m != i);
          ALPAKA_ASSERT_ACC(int(y[m]) - int(y[i]) >= 0);
          ALPAKA_ASSERT_ACC(int(y[m]) - int(y[i]) <= 1);
          if (std::abs(int(x[m]) - int(x[i])) <= 1) {
            auto l = nnn[k][jEquivalentClass]++;
            ALPAKA_ASSERT_ACC(l < maxNeighbours);
            nn[k][jEquivalentClass][l] = *p;
          }
        }
      });

      // for each pixel, look at all the pixels until the end of the module;
      // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
      // after the loop, all the pixel in each cluster should have the id equeal to the lowest
      // pixel in the cluster ( clus[i] == i ).
      bool more = true;
      int nloops = 0;
      while (alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, more)) {
        if (1 == nloops % 2) {
          cms::alpakatools::for_each_element_in_block_strided(acc, hist.size(), [&](uint32_t j) {
            auto p = hist.begin() + j;
            auto i = *p + firstPixel;
            auto m = clusterId[i];
            while (m != clusterId[m])
              m = clusterId[m];
            clusterId[i] = m;
          });
        } else {
          more = false;
          uint32_t k = 0u;
          cms::alpakatools::for_each_element_in_block_strided(acc, hist.size(), [&](uint32_t j) {
            k = j / blockDimension;
            const uint32_t jEquivalentClass = j % threadDimension;
            auto p = hist.begin() + j;
            auto i = *p + firstPixel;
            for (int kk = 0; kk < nnn[k][jEquivalentClass]; ++kk) {
              auto l = nn[k][jEquivalentClass][kk];
              auto m = l + firstPixel;
              ALPAKA_ASSERT_ACC(m != i);
              auto old = alpaka::atomicMin(acc, &clusterId[m], clusterId[i], alpaka::hierarchy::Blocks{});
              if (old != clusterId[i]) {
                // end the loop only if no changes were applied
                more = true;
              }
              alpaka::atomicMin(acc, &clusterId[i], old, alpaka::hierarchy::Blocks{});
            }  // nnloop
          });  // pixel loop
        }
        ++nloops;
      }  // end while

#ifdef GPU_DEBUG
      {
        auto& n0 = alpaka::declareSharedVar<int, __COUNTER__>(acc);
        if (threadIdxLocal == 0)
          n0 = nloops;
        alpaka::syncBlockThreads(acc);
#ifndef NDEBUG
        [[maybe_unused]] auto ok = n0 == nloops;
        ALPAKA_ASSERT_ACC(alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, ok));
#endif
        if (thisModuleId % 100 == 1)
          if (threadIdxLocal == 0)
            printf("# loops %d\n", nloops);
      }
#endif

      auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
      foundClusters = 0;
      alpaka::syncBlockThreads(acc);

      // find the number of different clusters, identified by a pixels with clus[i] == i;
      // mark these pixels with a negative id.
      cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
        if (id[i] != InvId) {  // skip invalid pixels
          if (clusterId[i] == static_cast<int>(i)) {
            auto old = alpaka::atomicInc(acc, &foundClusters, 0xffffffff, alpaka::hierarchy::Threads{});
            clusterId[i] = -(old + 1);
          }
        }
      });
      alpaka::syncBlockThreads(acc);

      // propagate the negative id to all the pixels in the cluster.
      cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
        if (id[i] != InvId) {  // skip invalid pixels
          if (clusterId[i] >= 0) {
            // mark each pixel in a cluster with the same id as the first one
            clusterId[i] = clusterId[clusterId[i]];
          }
        }
      });
      alpaka::syncBlockThreads(acc);

      // adjust the cluster id to be a positive value starting from 0
      cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
        if (id[i] == InvId) {  // skip invalid pixels
          clusterId[i] = -9999;
        } else {
          clusterId[i] = -clusterId[i] - 1;
        }
      });
      alpaka::syncBlockThreads(acc);

      if (threadIdxLocal == 0) {
        nClustersInModule[thisModuleId] = foundClusters;
        moduleId[blockIdx] = thisModuleId;
#ifdef GPU_DEBUG
        if (foundClusters > gMaxHit<TAcc>) {
          gMaxHit<TAcc> = foundClusters;
          if (foundClusters > 8)
            printf("max hit %d in %d\n", foundClusters, thisModuleId);
        }
#endif
#ifdef GPU_DEBUG
        if (thisModuleId % 100 == 1)
          printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
      }
    }
  };

}  // namespace gpuClustering

#endif  // plugin_SiPixelClusterizer_alpaka_gpuClustering_h
