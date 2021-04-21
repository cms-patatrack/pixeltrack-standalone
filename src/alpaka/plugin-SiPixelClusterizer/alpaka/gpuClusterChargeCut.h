#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include "AlpakaCore/alpakaKernelCommon.h"
#include "AlpakaCore/prefixScan.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"

namespace gpuClustering {

  struct clusterChargeCut {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(
        const T_Acc& acc,
        uint16_t* __restrict__ id,                 // module id of each pixel (modified if bad cluster)
        uint16_t const* __restrict__ adc,          //  charge of each pixel
        uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
        uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
        uint32_t const* __restrict__ moduleId,     // module id of each module
        int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
        const uint32_t numElements) const {
      const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      if (blockIdx >= moduleStart[0])
        return;

      auto firstPixel = moduleStart[1 + blockIdx];
      auto thisModuleId = id[firstPixel];
      assert(thisModuleId < MaxNumModules);
      assert(thisModuleId == moduleId[blockIdx]);

      auto nclus = nClustersInModule[thisModuleId];
      if (nclus == 0)
        return;

      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      if (threadIdxLocal == 0 && nclus > MaxNumClustersPerModules)
        printf("Warning too many clusters in module %d in block %d: %d > %d\n",
               thisModuleId,
               blockIdx,
               nclus,
               MaxNumClustersPerModules);

      // Stride = block size.
      const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

      // Get thread / CPU element indices in block.
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
          cms::alpakatools::element_index_range_in_block(acc, firstPixel);

      if (nclus > MaxNumClustersPerModules) {
        uint32_t firstElementIdx = firstElementIdxNoStride;
        uint32_t endElementIdx = endElementIdxNoStride;
        // remove excess  FIXME find a way to cut charge first....
        for (uint32_t i = firstElementIdx; i < numElements; ++i) {
          if (!cms::alpakatools::next_valid_element_index_strided(
                  i, firstElementIdx, endElementIdx, blockDimension, numElements))
            break;
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
        if (threadIdxLocal == 0)
          printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx);
#endif

      auto& charge = alpaka::declareSharedVar<int32_t[MaxNumClustersPerModules], __COUNTER__>(acc);
      auto& ok = alpaka::declareSharedVar<uint8_t[MaxNumClustersPerModules], __COUNTER__>(acc);
      auto& newclusId = alpaka::declareSharedVar<uint16_t[MaxNumClustersPerModules], __COUNTER__>(acc);

      assert(nclus <= MaxNumClustersPerModules);
      cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) { charge[i] = 0; });
      alpaka::syncBlockThreads(acc);

      uint32_t firstElementIdx = firstElementIdxNoStride;
      uint32_t endElementIdx = endElementIdxNoStride;
      for (uint32_t i = firstElementIdx; i < numElements; ++i) {
        if (!cms::alpakatools::next_valid_element_index_strided(
                i, firstElementIdx, endElementIdx, blockDimension, numElements))
          break;
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        alpaka::atomicOp<alpaka::AtomicAdd>(acc, &charge[clusterId[i]], static_cast<int32_t>(adc[i]));
      }
      alpaka::syncBlockThreads(acc);

      auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
      cms::alpakatools::for_each_element_in_block_strided(
          acc, nclus, [&](uint32_t i) { newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0; });
      alpaka::syncBlockThreads(acc);

      // renumber
      auto& ws = alpaka::declareSharedVar<uint16_t[32], __COUNTER__>(acc);
      cms::alpakatools::blockPrefixScan(acc, newclusId, nclus, ws);

      assert(nclus >= newclusId[nclus - 1]);

      if (nclus == newclusId[nclus - 1])
        return;

      nClustersInModule[thisModuleId] = newclusId[nclus - 1];
      alpaka::syncBlockThreads(acc);

      // mark bad cluster again
      cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) {
        if (0 == ok[i])
          newclusId[i] = InvId + 1;
      });
      alpaka::syncBlockThreads(acc);

      // reassign id
      firstElementIdx = firstElementIdxNoStride;
      endElementIdx = endElementIdxNoStride;
      for (uint32_t i = firstElementIdx; i < numElements; ++i) {
        if (!cms::alpakatools::next_valid_element_index_strided(
                i, firstElementIdx, endElementIdx, blockDimension, numElements))
          break;
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
  };

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
