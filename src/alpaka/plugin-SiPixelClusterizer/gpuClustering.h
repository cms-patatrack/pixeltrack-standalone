#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <cstdint>
#include <cstdio>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/HistoContainer.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"
#include "Geometry/phase1PixelTopology.h"

#define GPU_DEBUG true

namespace gpuClustering {

#ifdef GPU_DEBUG
  ALPAKA_STATIC_ACC_MEM_GLOBAL uint32_t gMaxHit = 0;
#endif

  struct countModules {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc& acc,
				  uint16_t const* __restrict__ id,
				  uint32_t* __restrict__ moduleStart,
				  int32_t* __restrict__ clusterId,
				  const unsigned int numElements) const {
      const uint32_t gridDimension(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u]);
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] = cms::alpakatools::element_global_index_range(acc);
      for (uint32_t threadIdx = firstElementIdxNoStride[0u], endElementIdx = endElementIdxNoStride[0u]; threadIdx < numElements;
	   threadIdx += gridDimension, endElementIdx += gridDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, numElements); ++i) {

	  clusterId[i] = i;
	  if (InvId == id[i])
	    continue;
	  int j = i - 1;
	  while (j >= 0 and id[j] == InvId)
	    --j;
	  if (j < 0 or id[j] != id[i]) {
	    // boundary...
            //auto loc = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, moduleStart, MaxNumModules);   
	    //auto loc = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, moduleStart, 2000u);
	    auto loc = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, &moduleStart[0], 1u);  // TO DO: does that work the same???????
	    assert(moduleStart[0] < MaxNumModules);

	    moduleStart[loc + 1] = i;
	  }
	}
      }

    }
  };

  //  __launch_bounds__(256,4)
  struct findClus {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc& acc,
				  uint16_t const* __restrict__ id,           // module id of each pixel
				  uint16_t const* __restrict__ x,            // local coordinates of each pixel
				  uint16_t const* __restrict__ y,            //
				  uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
				  uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
				  uint32_t* __restrict__ moduleId,           // output: module id of each module
				  int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
				  const unsigned int numElements) const {
      const uint32_t blockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      if (blockIdx >= moduleStart[0])
	return;

      auto firstPixel = moduleStart[1 + blockIdx];
      auto thisModuleId = id[firstPixel];
      assert(thisModuleId < MaxNumModules);

      const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] = cms::alpakatools::element_global_index_range(acc);

#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
	if (firstElementIdxNoStride[0] == 0)
	  printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx);
#endif

      const auto first = firstPixel + firstElementIdxNoStride[0];
      const auto end = firstPixel + endElementIdxNoStride[0u];

      // find the index of the first pixel not belonging to this module (or invalid)
      auto&& msize = alpaka::block::shared::st::allocVar<unsigned int, __COUNTER__>(acc);
      msize = numElements;
      alpaka::block::sync::syncBlockThreads(acc);

      // skip threads not associated to an existing pixel
      bool goOn = true;
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < numElements && goOn == true;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, numElements) && goOn == true; ++i) {
	  if (id[i] == InvId)  // skip invalid pixels
	    continue;
	  if (id[i] != thisModuleId) {  // find the first pixel in a different module
	    alpaka::atomic::atomicOp<alpaka::atomic::op::Min>(acc, &msize, i);
	    goOn = false;
	  }
	}
      }

      //init hist  (ymax=416 < 512 : 9bits)
      constexpr uint32_t maxPixInModule = 4000;
      constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
      using Hist = cms::alpakatools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
      auto&& hist = alpaka::block::shared::st::allocVar<Hist, __COUNTER__>(acc);
      //auto&& ws = alpaka::block::shared::st::allocVar<(typename Hist::Counter)[32], __COUNTER__>(acc);                 // TO DO: how to deal with typename??
      auto&& ws = alpaka::block::shared::st::allocVar<Hist::Counter[32], __COUNTER__>(acc);                 // TO DO: how to deal with typename??

      for (uint32_t threadIdx = firstElementIdxNoStride[0], endElementIdx = endElementIdxNoStride[0u]; threadIdx < Hist::totbins();
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t j = threadIdx; j < std::min(endElementIdx, Hist::totbins()); ++j) {
	  hist.off[j] = 0;
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);

      assert((msize == numElements) or ((msize < numElements) and (id[msize] != thisModuleId)));

      // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
      if (0 == firstElementIdxNoStride[0]) {
	if (msize - firstPixel > maxPixInModule) {
	  printf("too many pixels in module %d: %d > %d\n", thisModuleId, msize - firstPixel, maxPixInModule);
	  msize = maxPixInModule + firstPixel;
	}
      }

      alpaka::block::sync::syncBlockThreads(acc);
      assert(msize - firstPixel <= maxPixInModule);

#ifdef GPU_DEBUG
      auto&& totGood = alpaka::block::shared::st::allocVar<uint32_t, __COUNTER__>(acc);
      totGood = 0;
      alpaka::block::sync::syncBlockThreads(acc);
#endif

      // fill histo
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < msize;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, msize); ++i) {
	  if (id[i] == InvId)  // skip invalid pixels
	    continue;
	  hist.count(acc, y[i]);
#ifdef GPU_DEBUG
	  alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &totGood, 1u);
#endif
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);
      for (uint32_t i = firstElementIdxNoStride[0]; i < std::min(endElementIdxNoStride[0], 32u); ++i) {
	ws[i] = 0;  // used by prefix scan...
      }
      alpaka::block::sync::syncBlockThreads(acc);
      hist.finalize(acc, ws);
      alpaka::block::sync::syncBlockThreads(acc);
#ifdef GPU_DEBUG
      assert(hist.size() == totGood);
      if (thisModuleId % 100 == 1)
	if (firstElementIdxNoStride[0] == 0)
	  printf("histo size %d\n", hist.size());
#endif
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < msize;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, msize); ++i) {
	  if (id[i] == InvId)  // skip invalid pixels
	    continue;
	  hist.fill(acc, y[i], i - firstPixel);
	}
      }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      // assume that we can cover the whole module with up to 16 blockDimension-wide iterations
      constexpr int maxiter = 16;
#else
      auto maxiter = hist.size();
#endif
      // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
      constexpr int maxNeighbours = 10;
      assert((hist.size() / blockDimension) <= maxiter);
      // nearest neighbour
      uint16_t nn[maxiter][maxNeighbours];
      uint8_t nnn[maxiter];  // number of nn
      for (uint32_t k = 0; k < maxiter; ++k)
	nnn[k] = 0;

      alpaka::block::sync::syncBlockThreads(acc);  // for hit filling!

#ifdef GPU_DEBUG
      // look for anomalous high occupancy
      auto&& n40 = alpaka::block::shared::st::allocVar<uint32_t, __COUNTER__>(acc);
      auto&& n60 = alpaka::block::shared::st::allocVar<uint32_t, __COUNTER__>(acc);
      n40 = n60 = 0;
      alpaka::block::sync::syncBlockThreads(acc);
      for (uint32_t threadIdx = firstElementIdxNoStride[0], endElementIdx = endElementIdxNoStride[0]; threadIdx < Hist::nbins();
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t j = threadIdx; j < std::min(endElementIdx, Hist::nbins()); ++j) {
	  if (hist.size(j) > 60)
	    alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &n60, 1u);
	  if (hist.size(j) > 40)
	    alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &n40, 1u);
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);
      if (0 == firstElementIdxNoStride[0]) {
	if (n60 > 0)
	  printf("columns with more than 60 px %d in %d\n", n60, thisModuleId);
	else if (n40 > 0)
	  printf("columns with more than 40 px %d in %d\n", n40, thisModuleId);
      }
      alpaka::block::sync::syncBlockThreads(acc);
#endif

      // fill NN
      for (uint32_t threadIdx = firstElementIdxNoStride[0], endElementIdx = endElementIdxNoStride[0], k = 0U; threadIdx < hist.size();
	   threadIdx += blockDimension, endElementIdx += blockDimension, ++k) {
	for (uint32_t j = threadIdx; j < std::min(endElementIdx, hist.size()); ++j) {
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
      }

      // for each pixel, look at all the pixels until the end of the module;
      // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
      // after the loop, all the pixel in each cluster should have the id equeal to the lowest
      // pixel in the cluster ( clus[i] == i ).
      bool more = true;
      int nloops = 0;     
      while (alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, more)) {
	if (1 == nloops % 2) {
	  for (uint32_t threadIdx = firstElementIdxNoStride[0], endElementIdx = endElementIdxNoStride[0], k = 0U; threadIdx < hist.size();
	       threadIdx += blockDimension, endElementIdx += blockDimension, ++k) {
	    for (uint32_t j = threadIdx; j < std::min(endElementIdx, hist.size()); ++j) {
	      auto p = hist.begin() + j;
	      auto i = *p + firstPixel;
	      auto m = clusterId[i];
	      while (m != clusterId[m])
		m = clusterId[m];
	      clusterId[i] = m;
	    }
	  }
	} else {
	  more = false;
	  for (uint32_t threadIdx = firstElementIdxNoStride[0], endElementIdx = endElementIdxNoStride[0], k = 0U; threadIdx < hist.size();
	       threadIdx += blockDimension, endElementIdx += blockDimension, ++k) {
	    for (uint32_t j = threadIdx; j < std::min(endElementIdx, hist.size()); ++j) {
	      auto p = hist.begin() + j;
	      auto i = *p + firstPixel;
	      for (int kk = 0; kk < nnn[k]; ++kk) {
		auto l = nn[k][kk];
		auto m = l + firstPixel;
		assert(m != i);
		auto old = alpaka::atomic::atomicOp<alpaka::atomic::op::Min>(acc, &clusterId[m], clusterId[i]);
		if (old != clusterId[i]) {
		  // end the loop only if no changes were applied
		  more = true;
		}
		alpaka::atomic::atomicOp<alpaka::atomic::op::Min>(acc, &clusterId[i], old);
	      }  // nnloop
	    }    // pixel loop
	  }
	}
	++nloops;
      }  // end while

#ifdef GPU_DEBUG
      {
	auto&& n0 = alpaka::block::shared::st::allocVar<int, __COUNTER__>(acc);
	if (firstElementIdxNoStride[0] == 0)
	  n0 = nloops;
	alpaka::block::sync::syncBlockThreads(acc);
	auto ok = n0 == nloops;
	assert(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, ok));
	if (thisModuleId % 100 == 1)
	  if (firstElementIdxNoStride[0] == 0)
	    printf("# loops %d\n", nloops);
      }
#endif

      auto&& foundClusters = alpaka::block::shared::st::allocVar<unsigned int, __COUNTER__>(acc);
      foundClusters = 0;
      alpaka::block::sync::syncBlockThreads(acc);

      // find the number of different clusters, identified by a pixels with clus[i] == i;
      // mark these pixels with a negative id.
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < msize;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, msize); ++i) {
	  if (id[i] == InvId)  // skip invalid pixels
	    continue;
	  if (clusterId[i] == static_cast<int>(i)) {
	    auto old = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, &foundClusters, 0xffffffff);                    // TO DO: What the hell is this??
	    //auto old = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, &foundClusters, 4294967295u);
	    clusterId[i] = -(old + 1);
	  }
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);

      // propagate the negative id to all the pixels in the cluster.
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < msize;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, msize); ++i) {
	  if (id[i] == InvId)  // skip invalid pixels
	    continue;
	  if (clusterId[i] >= 0) {
	    // mark each pixel in a cluster with the same id as the first one
	    clusterId[i] = clusterId[clusterId[i]];
	  }
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);

      // adjust the cluster id to be a positive value starting from 0
      for (uint32_t threadIdx = first, endElementIdx = end; threadIdx < msize;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, msize); ++i) {
	  if (id[i] == InvId) {  // skip invalid pixels
	    clusterId[i] = -9999;
	    continue;
	  }
	  clusterId[i] = -clusterId[i] - 1;
	}
      }
      alpaka::block::sync::syncBlockThreads(acc);

      if (firstElementIdxNoStride[0] == 0) {
	nClustersInModule[thisModuleId] = foundClusters;
	moduleId[blockIdx] = thisModuleId;
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
      }
    }
  };

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
