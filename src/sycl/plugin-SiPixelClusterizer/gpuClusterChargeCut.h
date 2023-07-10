#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>

#include "SYCLCore/syclAtomic.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/prefixScan.h"
#include "SYCLCore/printf.h"

#include "SYCLDataFormats/gpuClusteringConstants.h"

namespace gpuClustering {

  void clusterChargeCut(uint16_t* __restrict__ id,                 // module id of each pixel (modified if bad cluster)
                        uint16_t const* __restrict__ adc,          //  charge of each pixel
                        uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                        uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
                        uint32_t const* __restrict__ moduleId,     // module id of each module
                        int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
                        uint32_t numElements,
                        sycl::nd_item<1> item) {
    if (item.get_group(0) >= moduleStart[0])
      return;

    auto firstPixel = moduleStart[1 + item.get_group(0)];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < MaxNumModules);
    assert(thisModuleId == moduleId[item.get_group(0)]);

    auto nclus = nClustersInModule[thisModuleId];
    if (nclus == 0)
      return;

    if (item.get_local_id(0) == 0 && nclus > MaxNumClustersPerModules) {
      printf("Warning too many clusters in module %d in block %d: %d > %d\n",
             thisModuleId,
             item.get_group(0),
             nclus,
             MaxNumClustersPerModules);
    }

    auto first = firstPixel + item.get_local_id(0);

    if (nclus > MaxNumClustersPerModules) {
      // remove excess  FIXME find a way to cut charge first....
      for (auto i = first; i < numElements; i += item.get_local_range(0)) {
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        if (clusterId[i] >= MaxNumClustersPerModules) {
          id[i] = InvId;
          clusterId[i] = InvId;
        }
      }
      nclus = MaxNumClustersPerModules;
    }

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(0) == 0)
        printf("start clusterizer for module %d in block %d\n", thisModuleId, item.get_group(0));
#endif

    auto chargebuff =
        sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[MaxNumClustersPerModules]>(item.get_group());
    int32_t* charge = (int32_t*)chargebuff.get();
    auto okbuff =
        sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[MaxNumClustersPerModules]>(item.get_group());
    uint8_t* ok = (uint8_t*)okbuff.get();
    auto newclusIdbuff =
        sycl::ext::oneapi::group_local_memory_for_overwrite<uint16_t[MaxNumClustersPerModules]>(item.get_group());
    uint16_t* newclusId = (uint16_t*)newclusIdbuff.get();

    assert(nclus <= MaxNumClustersPerModules);
    for (auto i = item.get_local_id(0); i < nclus; i += item.get_local_range(0)) {
      charge[i] = 0;
    }

    sycl::group_barrier(item.get_group());
    for (auto i = first; i < numElements; i += item.get_local_range(0)) {
      if (id[i] == InvId)
        continue;  // not valid
      if (id[i] != thisModuleId)
        break;  // end of module
      cms::sycltools::atomic_fetch_add<int32_t, sycl::access::address_space::local_space, sycl::memory_scope::work_group>(
          &charge[clusterId[i]], static_cast<int32_t>(adc[i]));
    }
    sycl::group_barrier(item.get_group());

    auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
    for (auto i = item.get_local_id(0); i < nclus; i += item.get_local_range(0)) {
      newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0;
    }

    sycl::group_barrier(item.get_group());

    // renumber
    auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint16_t[32]>(item.get_group());
    uint16_t* ws = (uint16_t*)wsbuff.get();
    cms::sycltools::blockPrefixScan(newclusId, nclus, item, ws);

    assert(nclus >= newclusId[nclus - 1]);

    if (nclus == newclusId[nclus - 1])
      return;

    nClustersInModule[thisModuleId] = newclusId[nclus - 1];
    sycl::group_barrier(item.get_group());

    // mark bad cluster again
    for (auto i = item.get_local_id(0); i < nclus; i += item.get_local_range(0)) {
      if (0 == ok[i])
        newclusId[i] = InvId + 1;
    }
    sycl::group_barrier(item.get_group());

    // reassign id
    for (auto i = first; i < numElements; i += item.get_local_range(0)) {
      if (id[i] == InvId)
        continue;  // not valid
      if (id[i] != thisModuleId)
        break;  // end of module
      clusterId[i] = newclusId[clusterId[i]] - 1;
      if (clusterId[i] == InvId)
        id[i] = InvId;
    }
    //done
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h