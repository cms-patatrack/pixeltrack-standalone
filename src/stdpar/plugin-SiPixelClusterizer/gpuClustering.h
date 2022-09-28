#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <iterator>
#include <ranges>

#include "Geometry/phase1PixelTopology.h"
#include "CUDACore/HistoContainer.h"
#include "CUDACore/portableAtomicOp.h"

#include "gpuClusteringConstants.h"

namespace gpuClustering {

#ifdef GPU_DEBUG
  uint32_t gMaxHit = 0;
#endif

  void countModules(uint16_t const* __restrict__ id,
                    uint32_t* __restrict__ moduleStart,
                    int32_t* __restrict__ clusterId,
                    int numElements) {
    auto iter{std::views::iota(0, numElements)};
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto i) {
      clusterId[i] = i;
      if (InvId != id[i]) {
        auto j = i - 1;
        while (j >= 0 and id[j] == InvId)
          --j;
        if (j < 0 or id[j] != id[i]) {
          // boundary...
          auto loc = cms::cuda::atomicInc(moduleStart, MaxNumModules);
          moduleStart[loc + 1] = i;
        }
      }
    });
  }

  void findClus(uint16_t const* __restrict__ id,           // module id of each pixel
                uint16_t const* __restrict__ x,            // local coordinates of each pixel
                uint16_t const* __restrict__ y,            //
                uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
                uint32_t* __restrict__ moduleId,           // output: module id of each module
                int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
                int numElements) {
    uint32_t firstModule = 0;
    uint32_t endModule = moduleStart[0];
    auto iter{std::views::iota(firstModule, endModule)};
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto module) {
      auto firstPixel = moduleStart[1 + module];
      auto thisModuleId = id[firstPixel];
      assert(thisModuleId < MaxNumModules);

#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
        printf("start clusterizer for module %d\n", thisModuleId);
#endif

      auto first = firstPixel;

      // find the index of the first pixel not belonging to this module (or invalid)
      int msize = numElements;

      // skip threads not associated to an existing pixel
      for (int i = first; i < numElements; i++) {
        if (id[i] == InvId)  // skip invalid pixels
          continue;
        if (id[i] != thisModuleId) {  // find the first pixel in a different module
          cms::cuda::atomicMin(&msize, i);
          break;
        }
      }

      //init hist  (ymax=416 < 512 : 9bits)
      constexpr uint32_t maxPixInModule = 4000;
      constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
      using Hist = cms::cuda::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
      Hist hist;
      for (auto j = 0; j < Hist::totbins(); ++j) {
        hist.off[j] = 0;
      }

      assert((msize == numElements) or ((msize < numElements) and (id[msize] != thisModuleId)));

      // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
      if (msize - firstPixel > maxPixInModule) {
        printf("too many pixels in module %d: %d > %d\n", thisModuleId, msize - firstPixel, maxPixInModule);
        msize = maxPixInModule + firstPixel;
      }

      assert(msize - firstPixel <= maxPixInModule);

#ifdef GPU_DEBUG
      uint32_t totGood;
      totGood = 0;
#endif

      // fill histo
      for (int i = first; i < msize; ++i) {
        if (id[i] == InvId)  // skip invalid pixels
          continue;
        hist.count(y[i]);
#ifdef GPU_DEBUG
        std::atomic_ref<uint32_t> inc{totGood};
        ++inc;
#endif
      }
      hist.finalizeSeq();
#ifdef GPU_DEBUG
      assert(hist.size() == totGood);
      if (thisModuleId % 100 == 1)
        printf("histo size %d\n", hist.size());
#endif
      for (int i = first; i < msize; ++i) {
        if (id[i] == InvId)  // skip invalid pixels
          continue;
        hist.fill(y[i], i - firstPixel);
      }

      auto maxiter = hist.size();
      // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
      constexpr int maxNeighbours = 10;
      assert(hist.size() <= maxiter);
      // nearest neighbour
      uint16_t nn[maxiter][maxNeighbours];
      uint8_t nnn[maxiter];  // number of nn
      for (uint32_t k = 0; k < maxiter; ++k)
        nnn[k] = 0;

#ifdef GPU_DEBUG
      // look for anomalous high occupancy
      uint32_t n40, n60;
      n40 = n60 = 0;
      for (auto j = 0; j < Hist::nbins(); ++j) {
        if (hist.size(j) > 60) {
          std::atomic_ref<uint32_t> inc{n60};
          ++inc;
        }
        if (hist.size(j) > 40) {
          std::atomic_ref<uint32_t> inc{n40};
          ++inc;
        }
      }

      if (n60 > 0)
        printf("columns with more than 60 px %d in %d\n", n60, thisModuleId);
      else if (n40 > 0)
        printf("columns with more than 40 px %d in %d\n", n40, thisModuleId);
#endif

      // fill NN
      for (uint32_t j = 0, k = 0U; j < hist.size(); ++j, ++k) {
        assert(k < maxiter);
        auto p = hist.begin() + j;
        auto i = *p + firstPixel;
        assert(id[i] != InvId);
        assert(id[i] == thisModuleId);  // same module
        int be = Hist::bin(y[i] + 1);
        auto e = hist.end(be);
        ++p;
        assert(0 == nnn[k]);
        for (; p < e; ++p) {
          auto m = (*p) + firstPixel;
          assert(m != i);
          assert(int(y[m]) - int(y[i]) >= 0);
          assert(int(y[m]) - int(y[i]) <= 1);
          if (std::abs(int(x[m]) - int(x[i])) > 1)
            continue;
          auto l = nnn[k]++;
          assert(l < maxNeighbours);
          nn[k][l] = *p;
        }
      }

      // for each pixel, look at all the pixels until the end of the module;
      // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
      // after the loop, all the pixel in each cluster should have the id equeal to the lowest
      // pixel in the cluster ( clus[i] == i ).
      bool more = true;
      int nloops = 0;
      while (more) {
        if (1 == nloops % 2) {
          for (uint32_t j = 0, k = 0U; j < hist.size(); ++j, ++k) {
            auto p = hist.begin() + j;
            auto i = *p + firstPixel;
            auto m = clusterId[i];
            while (m != clusterId[m])
              m = clusterId[m];
            clusterId[i] = m;
          }
        } else {
          more = false;
          for (uint32_t j = 0, k = 0U; j < hist.size(); ++j, ++k) {
            auto p = hist.begin() + j;
            auto i = *p + firstPixel;
            for (int kk = 0; kk < nnn[k]; ++kk) {
              auto l = nn[k][kk];
              auto m = l + firstPixel;
              assert(m != i);
              auto old = cms::cuda::atomicMin(&clusterId[m], clusterId[i]);
              if (old != clusterId[i]) {
                // end the loop only if no changes were applied
                more = true;
              }
              cms::cuda::atomicMin(&clusterId[i], old);
            }  // nnloop
          }    // pixel loop
        }
        ++nloops;
      }  // end while

#ifdef GPU_DEBUG
      {
        if (thisModuleId % 100 == 1)
          printf("# loops %d\n", nloops);
      }
#endif

      unsigned int foundClusters = 0;

      // find the number of different clusters, identified by a pixels with clus[i] == i;
      // mark these pixels with a negative id.
      for (int i = first; i < msize; ++i) {
        if (id[i] == InvId)  // skip invalid pixels
          continue;
        if (clusterId[i] == i) {
          auto old = cms::cuda::atomicInc(&foundClusters, 0xffffffff);
          clusterId[i] = -(old + 1);
        }
      }

      // propagate the negative id to all the pixels in the cluster.
      for (int i = first; i < msize; ++i) {
        if (id[i] == InvId)  // skip invalid pixels
          continue;
        if (clusterId[i] >= 0) {
          // mark each pixel in a cluster with the same id as the first one
          clusterId[i] = clusterId[clusterId[i]];
        }
      }

      // adjust the cluster id to be a positive value starting from 0
      for (int i = first; i < msize; ++i) {
        if (id[i] == InvId) {  // skip invalid pixels
          clusterId[i] = -9999;
          continue;
        }
        clusterId[i] = -clusterId[i] - 1;
      }

      nClustersInModule[thisModuleId] = foundClusters;
      moduleId[module] = thisModuleId;
#ifdef GPU_DEBUG
      if (foundClusters > gMaxHit) {
        gMaxHit = foundClusters;
        if (foundClusters > 8)
          printf("max hit %d in %d\n", foundClusters, thisModuleId);
      }
#endif
#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
        printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
    });
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
