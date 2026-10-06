#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include "CUDACore/cuda_assert.h"
#include "CUDACore/prefixScan.h"

#include "gpuClusteringConstants.h"

namespace gpuClustering {

  void clusterChargeCut(uint16_t* __restrict__ id,                 // module id of each pixel (modified if bad cluster)
                        uint16_t const* __restrict__ adc,          //  charge of each pixel
                        uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                        uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
                        uint32_t const* __restrict__ moduleId,     // module id of each module
                        int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
                        uint32_t numElements) {
    int32_t charge[MaxNumClustersPerModules];
    uint8_t ok[MaxNumClustersPerModules];
    uint16_t newclusId[MaxNumClustersPerModules];

    uint32_t firstModule = 0;
    auto endModule = moduleStart[0];
    for (auto module = firstModule; module < endModule; module += 1) {
      auto firstPixel = moduleStart[1 + module];
      auto thisModuleId = id[firstPixel];
      assert(thisModuleId < MaxNumModules);
      assert(thisModuleId == moduleId[module]);

      auto nclus = nClustersInModule[thisModuleId];
      if (nclus == 0)
        continue;

      if (nclus > MaxNumClustersPerModules)
        printf("Warning too many clusters in module %d in block %d: %d > %d\n",
               thisModuleId,
               0,
               nclus,
               MaxNumClustersPerModules);

      auto first = firstPixel;

      if (nclus > MaxNumClustersPerModules) {
        // remove excess  FIXME find a way to cut charge first....
        for (auto i = first; i < numElements; i++) {
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
        printf("start cluster charge cut for module %d in block %d\n", thisModuleId, 0);
#endif

      assert(nclus <= MaxNumClustersPerModules);
      for (uint32_t i = 0; i < nclus; i++) {
        charge[i] = 0;
      }

      for (auto i = first; i < numElements; i++) {
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        atomicAdd(&charge[clusterId[i]], adc[i]);
      }

      auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
      for (uint32_t i = 0; i < nclus; i++) {
        newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0;
      }

      // renumber

      cms::cuda::blockPrefixScan(newclusId, nclus);

      assert(nclus >= newclusId[nclus - 1]);

      if (nclus == newclusId[nclus - 1])
        continue;

      nClustersInModule[thisModuleId] = newclusId[nclus - 1];

      // mark bad cluster again
      for (uint32_t i = 0; i < nclus; i++) {
        if (0 == ok[i])
          newclusId[i] = InvId + 1;
      }

      // reassign id
      for (auto i = first; i < numElements; i++) {
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        clusterId[i] = newclusId[clusterId[i]] - 1;
        if (clusterId[i] == InvId)
          id[i] = InvId;
      }

      //done
    }  // loop on modules
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
