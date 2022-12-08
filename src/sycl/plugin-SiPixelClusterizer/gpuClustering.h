#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <CL/sycl.hpp>
#include <cstdint>
#include <cstdio>

#include "Geometry/phase1PixelTopology.h"
#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"
#include "SYCLCore/printf.h"

#include "SYCLDataFormats/gpuClusteringConstants.h"

// #define GPU_DEBUG

namespace gpuClustering {

  void countModules(uint16_t const* __restrict__ id,
                    uint32_t* __restrict__ moduleStart,
                    int32_t* __restrict__ clusterId,
                    int numElements,
                    sycl::nd_item<1> item) {
    int first = item.get_local_range(0) * item.get_group(0) + item.get_local_id(0);
    for (int i = first; i < numElements; i += item.get_group_range(0) * item.get_local_range(0)) {
      clusterId[i] = i;
      if (InvId == id[i])
        continue;
      auto j = i - 1;
      while (j >= 0 and id[j] == InvId)
        --j;
      if (j < 0 or id[j] != id[i]) {
        auto loc = cms::sycltools::
            atomic_fetch_compare_inc<uint32_t, sycl::access::address_space::global_space, sycl::memory_scope::device>(
                moduleStart, static_cast<uint32_t>(MaxNumModules));
        moduleStart[loc + 1] = i;
      }
    }
  }

  //init hist  (ymax=416 < 512 : 9bits)
  constexpr int maxPixInModule = 4000;
  constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
  using Hist = cms::sycltools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;

  void findClusGPU(uint16_t const* __restrict__ id,           // module id of each pixel
                   uint16_t const* __restrict__ x,            // local coordinates of each pixel
                   uint16_t const* __restrict__ y,            //
                   uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                   uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
                   uint32_t* __restrict__ moduleId,           // output: module id of each module
                   int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
                   int numElements,
                   sycl::nd_item<1> item) {
    if (item.get_group(0) >= moduleStart[0])
      return;

#ifdef GPU_DEBUG
    uint32_t gMaxHit = 0;  //FIXME_ think about a global accessor with access mode atomic maybe
#endif

    auto firstPixel = moduleStart[1 + item.get_group(0)];
    auto thisModuleId = id[firstPixel];

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(0) == 0)
        printf("start clusterizer for module %d in block %d\n", thisModuleId, item.get_group(0));
#endif

    auto first = firstPixel + item.get_local_id(0);

    // find the index of the first pixel not belonging to this module (or invalid)
    auto msizebuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
    int* msize = (int*)msizebuff.get();
    *msize = numElements;
    sycl::group_barrier(item.get_group());

    // skip threads not associated to an existing pixel
    for (int i = first; i < numElements; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (id[i] != thisModuleId) {  //find the first pixel in a different module
        cms::sycltools::atomic_fetch_min<int, sycl::access::address_space::local_space, sycl::memory_scope::work_group>(
            static_cast<int*>(msize), static_cast<int>(i));
        break;
      }
    }

    auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[32]>(item.get_group());
    uint32_t* ws = (uint32_t*)wsbuff.get();
    auto histbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<Hist>(item.get_group());
    Hist* hist = (Hist*)histbuff.get();

    //constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2
    for (auto j = item.get_local_id(0); j < Hist::totbins(); j += item.get_local_range(0)) {
      hist->off[j] = 0;
    }
    sycl::group_barrier(item.get_group());

    // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
    if (0 == item.get_local_id(0)) {
      if ((*msize - static_cast<int>(firstPixel)) > maxPixInModule) {
        printf("too many pixels in module %d: %d > %d\n",
               thisModuleId,
               *msize - static_cast<int>(firstPixel),
               maxPixInModule);
        *msize = maxPixInModule + firstPixel;
      }
    }
    sycl::group_barrier(item.get_group());

#ifdef GPU_DEBUG
    auto totGoodbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
    uint32_t* totGood = (uint32_t*)totGoodbuff.get();
    *totGood = 0;
    sycl::group_barrier(item.get_group());
#endif

    // fill histo
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId) {  // skip invalid pixels
        continue;
      }
      hist->count(y[i]);
#ifdef GPU_DEBUG
      cms::sycltools::atomic_fetch_add<uint32_t, sycl::access::address_space::local_space, sycl::memory_scope::device>(
          totGood, static_cast<uint32_t>(1));
#endif
    }

    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) < 32)
      ws[item.get_local_id(0)] = 0;  // used by prefix scan...

    sycl::group_barrier(item.get_group());
    hist->finalize(item, ws);
    sycl::group_barrier(item.get_group());

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(0) == 0)
        printf("histo size %d\n", hist->size());
#endif

    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      hist->fill(y[i], i - firstPixel);
    }

    // assume that we can cover the whole module with up to 16 blockDim.x-wide iterations -> this is true with blockDim.x=128
    // When the number of threads per block is changed, also this number must be changed to be still able to cover the whole module
    // when the bug with any_of_group will be fixed, the variables values can be restored.
    // set to 32 if ThreadsPerBlock is 32 (for CPU) -> SYCL_BUG_
    constexpr int maxiter = 16;  // in serial version: maxiter = hist->size()
    // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
    constexpr int maxNeighbours = 10;

    //nearest neighbour
    uint16_t nn[maxiter][maxNeighbours];
    uint8_t nnn[maxiter];  // number of nn
    assert((hist->size() / item.get_local_range(0)) <= maxiter);
    for (uint32_t k = 0; k < maxiter; ++k)
      nnn[k] = 0;
    sycl::group_barrier(item.get_group());  // for hit filling

#ifdef GPU_DEBUG
    // look for anomalous high occupancy
    auto n40buff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
    uint32_t* n40 = (uint32_t*)n40buff.get();
    auto n60buff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
    uint32_t* n60 = (uint32_t*)n60buff.get();
    *n40 = *n60 = 0;
    sycl::group_barrier(item.get_group());
    for (auto j = item.get_local_id(0); j < Hist::nbins(); j += item.get_local_range(0)) {
      if (hist->size(j) > 60)
        cms::sycltools::atomic_fetch_add<uint32_t, sycl::access::address_space::local_space, sycl::memory_scope::device>(
            n60, static_cast<uint32_t>(1));
      if (hist->size(j) > 40)
        cms::sycltools::atomic_fetch_add<uint32_t, sycl::access::address_space::local_space, sycl::memory_scope::device>(
            n40, static_cast<uint32_t>(1));
    }
    sycl::group_barrier(item.get_group());

    if (0 == item.get_local_id(0)) {
      if (*n60 > 0)
        printf("columns with more than 60 px %d in %d\n", *n60, thisModuleId);
      else if (*n40 > 0)
        printf("columns with more than 40 px %d in %d\n", *n40, thisModuleId);
    }
    sycl::group_barrier(item.get_group());
#endif

    //fill NN
    for (unsigned int j = item.get_local_id(0), k = 0U; j < hist->size(); j += item.get_local_range(0), ++k) {
      assert(k < maxiter);
      auto p = hist->begin() + j;
      auto i = *p + firstPixel;
      assert(id[i] != InvId);
      assert(id[i] == thisModuleId);  // same module
      int be = Hist::bin(y[i] + 1);
      auto e = hist->end(be);
      ++p;
      assert(0 == nnn[k]);
      for (; p < e; ++p) {
        auto m = (*p) + firstPixel;
        assert(m != i);
        assert(int(y[m]) - int(y[i]) >= 0);
        assert(int(y[m]) - int(y[i]) <= 1);
        if (sycl::abs(int(x[m]) - int(x[i])) > 1)
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

    while ((sycl::group_barrier(item.get_group()), sycl::any_of_group(item.get_group(), more))) {
      if (1 == nloops % 2) {
        for (unsigned int j = item.get_local_id(0), k = 0U; j < hist->size(); j += item.get_local_range(0), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          auto m = clusterId[i];
          while (m != clusterId[m])
            m = clusterId[m];
          clusterId[i] = m;
        }
      } else {
        more = false;
        for (unsigned int j = item.get_local_id(0), k = 0U; j < hist->size(); j += item.get_local_range(0), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          for (int kk = 0; kk < nnn[k]; ++kk) {
            auto l = nn[k][kk];
            auto m = l + firstPixel;
            assert(m != i);
            auto old = cms::sycltools::
                atomic_fetch_min<int32_t, sycl::access::address_space::global_space, sycl::memory_scope::device>(
                    static_cast<int32_t*>(&clusterId[m]), static_cast<int32_t>(clusterId[i]));

            if (old != clusterId[i]) {
              // end the loop only if no changes were applied
              more = true;
            }

            cms::sycltools::
                atomic_fetch_min<int32_t, sycl::access::address_space::global_space, sycl::memory_scope::device>(
                    static_cast<int32_t*>(&clusterId[i]), static_cast<int32_t>(old));
          }  // nnloop
        }    // pixel loop
      }
      ++nloops;
    }  // end while

#ifdef GPU_DEBUG
    {
      auto n0buff = sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
      int* n0 = (int*)n0buff.get();
      if (item.get_local_id(0) == 0)
        *n0 = nloops;
      sycl::group_barrier(item.get_group());
      if (thisModuleId % 100 == 1)
        if (item.get_local_id(0) == 0)
          printf("# loops %d\n", nloops);
    }
#endif

    auto foundClustersbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned int>(item.get_group());
    unsigned int* foundClusters = (unsigned int*)foundClustersbuff.get();
    *foundClusters = 0;

    sycl::group_barrier(item.get_group());
    // find the number of different clusters, identified by a pixels with clus[i] == i;
    // mark these pixels with a negative id.
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] == i) {
        auto old = cms::sycltools::
            atomic_fetch_add<unsigned int, sycl::access::address_space::local_space, sycl::memory_scope::work_group>(
                foundClusters, static_cast<unsigned int>(1));
        clusterId[i] = -(old + 1);
      }
    }
    sycl::group_barrier(item.get_group());
    // propagate the negative id to all the pixels in the cluster.
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] >= 0) {
        // mark each pixel in a cluster with the same id as the first one
        clusterId[i] = clusterId[clusterId[i]];
      }
    }
    sycl::group_barrier(item.get_group());

    // adjust the cluster id to be a positive value starting from 0
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId) {  // skip invalid pixels
        clusterId[i] = -9999;
        continue;
      }
      clusterId[i] = -clusterId[i] - 1;
    }
    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) == 0) {
      nClustersInModule[thisModuleId] = *foundClusters;
      moduleId[item.get_group(0)] = thisModuleId;
#ifdef GPU_DEBUG
      if (*foundClusters > gMaxHit) {
        gMaxHit = *foundClusters;
        if (*foundClusters > 8)
          printf("max hit %d in %d\n", *foundClusters, thisModuleId);
      }
      if (thisModuleId % 100 == 1)
        printf("%d clusters in module %d\n", *foundClusters, thisModuleId);
#endif
    }
  }

  void findClusCPU(uint16_t const* __restrict__ id,           // module id of each pixel
                   uint16_t const* __restrict__ x,            // local coordinates of each pixel
                   uint16_t const* __restrict__ y,            //
                   uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                   uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
                   uint32_t* __restrict__ moduleId,           // output: module id of each module
                   int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
                   int numElements,
                   sycl::nd_item<1> item) {
    if (item.get_group(0) >= moduleStart[0])
      return;

#ifdef GPU_DEBUG
    uint32_t gMaxHit = 0;  //FIXME_ think about a global accessor with access mode atomic maybe
#endif

    auto firstPixel = moduleStart[1 + item.get_group(0)];
    auto thisModuleId = id[firstPixel];

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(0) == 0)
        printf("start clusterizer for module %d in block %d\n", thisModuleId, item.get_group(0));
#endif

    auto first = firstPixel + item.get_local_id(0);

    // find the index of the first pixel not belonging to this module (or invalid)
    auto msizebuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
    int* msize = (int*)msizebuff.get();
    *msize = numElements;
    sycl::group_barrier(item.get_group());

    // skip threads not associated to an existing pixel
    for (int i = first; i < numElements; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (id[i] != thisModuleId) {  //find the first pixel in a different module
        cms::sycltools::atomic_fetch_min<int, sycl::access::address_space::local_space, sycl::memory_scope::work_group>(
            static_cast<int*>(msize), static_cast<int>(i));
        break;
      }
    }

    auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[32]>(item.get_group());
    uint32_t* ws = (uint32_t*)wsbuff.get();
    auto histbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<Hist>(item.get_group());
    Hist* hist = (Hist*)histbuff.get();

    //constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2
    for (auto j = item.get_local_id(0); j < Hist::totbins(); j += item.get_local_range(0)) {
      hist->off[j] = 0;
    }
    sycl::group_barrier(item.get_group());

    // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
    if (0 == item.get_local_id(0)) {
      if ((*msize - static_cast<int>(firstPixel)) > maxPixInModule) {
        printf("too many pixels in module %d: %d > %d\n",
               thisModuleId,
               *msize - static_cast<int>(firstPixel),
               maxPixInModule);
        *msize = maxPixInModule + firstPixel;
      }
    }
    sycl::group_barrier(item.get_group());

#ifdef GPU_DEBUG
    auto totGoodbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
    uint32_t* totGood = (uint32_t*)totGoodbuff.get();
    *totGood = 0;
    sycl::group_barrier(item.get_group());
#endif

    // fill histo
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId) {  // skip invalid pixels
        continue;
      }
      hist->count(y[i]);
#ifdef GPU_DEBUG
      cms::sycltools::atomic_fetch_add<uint32_t, sycl::access::address_space::local_space, sycl::memory_scope::device>(
          totGood, static_cast<uint32_t>(1));
#endif
    }

    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) < 32)
      ws[item.get_local_id(0)] = 0;  // used by prefix scan...

    sycl::group_barrier(item.get_group());
    hist->finalize(item, ws);
    sycl::group_barrier(item.get_group());

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(0) == 0)
        printf("histo size %d\n", hist->size());
#endif

    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      hist->fill(y[i], i - firstPixel);
    }

    // assume that we can cover the whole module with up to 16 blockDim.x-wide iterations -> this is true with blockDim.x=128
    // When the number of threads per block is changed, also this number must be changed to be still able to cover the whole module
    // when the bug with any_of_group will be fixed, the variables values can be restored.
    // set to 32 if ThreadsPerBlock is 32 (for CPU) -> SYCL_BUG_
    constexpr int maxiter = 32;  // in serial version: maxiter = hist->size()
    // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
    constexpr int maxNeighbours = 10;

    //nearest neighbour
    uint16_t nn[maxiter][maxNeighbours];
    uint8_t nnn[maxiter];  // number of nn
    assert((hist->size() / item.get_local_range(0)) <= maxiter);
    for (uint32_t k = 0; k < maxiter; ++k)
      nnn[k] = 0;
    sycl::group_barrier(item.get_group());  // for hit filling

#ifdef GPU_DEBUG
    // look for anomalous high occupancy
    auto n40buff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
    uint32_t* n40 = (uint32_t*)n40buff.get();
    auto n60buff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
    uint32_t* n60 = (uint32_t*)n60buff.get();
    *n40 = *n60 = 0;
    sycl::group_barrier(item.get_group());
    for (auto j = item.get_local_id(0); j < Hist::nbins(); j += item.get_local_range(0)) {
      if (hist->size(j) > 60)
        cms::sycltools::atomic_fetch_add<uint32_t, sycl::access::address_space::local_space, sycl::memory_scope::device>(
            n60, static_cast<uint32_t>(1));
      if (hist->size(j) > 40)
        cms::sycltools::atomic_fetch_add<uint32_t, sycl::access::address_space::local_space, sycl::memory_scope::device>(
            n40, static_cast<uint32_t>(1));
    }
    sycl::group_barrier(item.get_group());

    if (0 == item.get_local_id(0)) {
      if (*n60 > 0)
        printf("columns with more than 60 px %d in %d\n", *n60, thisModuleId);
      else if (*n40 > 0)
        printf("columns with more than 40 px %d in %d\n", *n40, thisModuleId);
    }
    sycl::group_barrier(item.get_group());
#endif

    //fill NN
    for (unsigned int j = item.get_local_id(0), k = 0U; j < hist->size(); j += item.get_local_range(0), ++k) {
      assert(k < maxiter);
      auto p = hist->begin() + j;
      auto i = *p + firstPixel;
      assert(id[i] != InvId);
      assert(id[i] == thisModuleId);  // same module
      int be = Hist::bin(y[i] + 1);
      auto e = hist->end(be);
      ++p;
      assert(0 == nnn[k]);
      for (; p < e; ++p) {
        auto m = (*p) + firstPixel;
        assert(m != i);
        assert(int(y[m]) - int(y[i]) >= 0);
        assert(int(y[m]) - int(y[i]) <= 1);
        if (sycl::abs(int(x[m]) - int(x[i])) > 1)
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

    while ((sycl::group_barrier(item.get_group()), sycl::any_of_group(item.get_group(), more))) {
      if (1 == nloops % 2) {
        for (unsigned int j = item.get_local_id(0), k = 0U; j < hist->size(); j += item.get_local_range(0), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          auto m = clusterId[i];
          while (m != clusterId[m])
            m = clusterId[m];
          clusterId[i] = m;
        }
      } else {
        more = false;
        for (unsigned int j = item.get_local_id(0), k = 0U; j < hist->size(); j += item.get_local_range(0), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          for (int kk = 0; kk < nnn[k]; ++kk) {
            auto l = nn[k][kk];
            auto m = l + firstPixel;
            assert(m != i);
            auto old = cms::sycltools::
                atomic_fetch_min<int32_t, sycl::access::address_space::global_space, sycl::memory_scope::device>(
                    static_cast<int32_t*>(&clusterId[m]), static_cast<int32_t>(clusterId[i]));

            if (old != clusterId[i]) {
              // end the loop only if no changes were applied
              more = true;
            }

            cms::sycltools::
                atomic_fetch_min<int32_t, sycl::access::address_space::global_space, sycl::memory_scope::device>(
                    static_cast<int32_t*>(&clusterId[i]), static_cast<int32_t>(old));
          }  // nnloop
        }    // pixel loop
      }
      ++nloops;
    }  // end while

#ifdef GPU_DEBUG
    {
      auto n0buff = sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
      int* n0 = (int*)n0buff.get();
      if (item.get_local_id(0) == 0)
        *n0 = nloops;
      sycl::group_barrier(item.get_group());
      if (thisModuleId % 100 == 1)
        if (item.get_local_id(0) == 0)
          printf("# loops %d\n", nloops);
    }
#endif

    auto foundClustersbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned int>(item.get_group());
    unsigned int* foundClusters = (unsigned int*)foundClustersbuff.get();
    *foundClusters = 0;

    sycl::group_barrier(item.get_group());
    // find the number of different clusters, identified by a pixels with clus[i] == i;
    // mark these pixels with a negative id.
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] == i) {
        auto old = cms::sycltools::
            atomic_fetch_add<unsigned int, sycl::access::address_space::local_space, sycl::memory_scope::work_group>(
                foundClusters, static_cast<unsigned int>(1));
        clusterId[i] = -(old + 1);
      }
    }
    sycl::group_barrier(item.get_group());
    // propagate the negative id to all the pixels in the cluster.
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] >= 0) {
        // mark each pixel in a cluster with the same id as the first one
        clusterId[i] = clusterId[clusterId[i]];
      }
    }
    sycl::group_barrier(item.get_group());

    // adjust the cluster id to be a positive value starting from 0
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId) {  // skip invalid pixels
        clusterId[i] = -9999;
        continue;
      }
      clusterId[i] = -clusterId[i] - 1;
    }
    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) == 0) {
      nClustersInModule[thisModuleId] = *foundClusters;
      moduleId[item.get_group(0)] = thisModuleId;
#ifdef GPU_DEBUG
      if (*foundClusters > gMaxHit) {
        gMaxHit = *foundClusters;
        if (*foundClusters > 8)
          printf("max hit %d in %d\n", *foundClusters, thisModuleId);
      }
      if (thisModuleId % 100 == 1)
        printf("%d clusters in module %d\n", *foundClusters, thisModuleId);
#endif
    }
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
