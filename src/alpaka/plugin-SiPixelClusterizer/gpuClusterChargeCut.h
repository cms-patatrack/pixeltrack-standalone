#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/prefixScan.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"

namespace gpuClustering {

  struct clusterChargeCut {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc& acc,
				  uint16_t* __restrict__ id,                 // module id of each pixel (modified if bad cluster)
				  uint16_t const* __restrict__ adc,          //  charge of each pixel
				  uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
				  uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
				  uint32_t const* __restrict__ moduleId,     // module id of each module
				  int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
				  const uint32_t numElements) const {

      const uint32_t blockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      if (blockIdx >= moduleStart[0])
	return;

      auto firstPixel = moduleStart[1 + blockIdx];
      auto thisModuleId = id[firstPixel];
      assert(thisModuleId < MaxNumModules);
      assert(thisModuleId == moduleId[blockIdx]);

      auto nclus = nClustersInModule[thisModuleId];
      if (nclus == 0)
	return;

      const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] = cms::alpakatools::element_global_index_range(acc);

      if (firstElementIdxNoStride[0] == 0 && nclus > MaxNumClustersPerModules)
	printf("Warning too many clusters in module %d in block %d: %d > %d\n",
	       thisModuleId,
	       blockIdx,
	       nclus,
	       MaxNumClustersPerModules);


      const auto first = firstPixel + firstElementIdxNoStride[0u];
      const auto end = firstPixel + endElementIdxNoStride[0u];

      if (nclus > MaxNumClustersPerModules) {

        bool goOn = true;
	for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < numElements && goOn == true;
	     threadIdx += blockDimension, endElementIdx += blockDimension) {
	  for (uint32_t i = threadIdx; i < std::min(endElementIdx, numElements) && goOn == true; ++i) {

	    if (id[i] == InvId)
	      continue;  // not valid
	    if (id[i] != thisModuleId)
	      goOn = false;  // end of module
	    if (clusterId[i] >= MaxNumClustersPerModules) {
	      id[i] = InvId;
	      clusterId[i] = InvId;
	    }

	  }
	}

	nclus = MaxNumClustersPerModules;
      }

      
#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
	if (firstElementIdxNoStride[0u] == 0)
	  printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx);
#endif

      auto&& charge = alpaka::block::shared::st::allocVar<int32_t[MaxNumClustersPerModules], __COUNTER__>(acc);
      auto&& ok = alpaka::block::shared::st::allocVar<uint8_t[MaxNumClustersPerModules], __COUNTER__>(acc);
      auto&& newclusId = alpaka::block::shared::st::allocVar<uint16_t[MaxNumClustersPerModules], __COUNTER__>(acc);

      assert(nclus <= MaxNumClustersPerModules);
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < nclus;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, nclus); ++i) {
	  charge[i] = 0;
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);

      bool goOn = true;
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < numElements && goOn == true;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, numElements) && goOn == true; ++i) {
	  if (id[i] == InvId)
	    continue;  // not valid
	  if (id[i] != thisModuleId)
	    goOn = false;  // end of module
	  alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &charge[clusterId[i]], static_cast<int32_t>(adc[i]));
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);

      auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < nclus;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, nclus); ++i) {
	  newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0;
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);

      // renumber
      auto&& ws = alpaka::block::shared::st::allocVar<uint16_t[32], __COUNTER__>(acc);
      cms::alpakatools::blockPrefixScan(acc, newclusId, nclus, ws);

      assert(nclus >= newclusId[nclus - 1]);

      if (nclus == newclusId[nclus - 1])
	return;

      nClustersInModule[thisModuleId] = newclusId[nclus - 1];
      alpaka::block::sync::syncBlockThreads(acc);

      // mark bad cluster again
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < nclus;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, nclus); ++i) {
	  if (0 == ok[i])
	    newclusId[i] = InvId + 1;
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);

      // reassign id
      goOn = true;
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < numElements && goOn == true;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, numElements) && goOn == true; ++i) {
	  if (id[i] == InvId)
	    continue;  // not valid
	  if (id[i] != thisModuleId)
	    goOn = false;  // end of module
	  clusterId[i] = newclusId[clusterId[i]] - 1;
	  if (clusterId[i] == InvId)
	    id[i] = InvId;
	}
      }

      //done
    }
  };

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
