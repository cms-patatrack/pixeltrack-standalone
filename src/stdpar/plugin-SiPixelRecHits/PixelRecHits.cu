// C++ headers
#include <algorithm>
#include <numeric>
#include <execution>
#include <ranges>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "CUDACore/cudaCheck.h"
#include "plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.h"  // !
#include "plugin-SiPixelClusterizer/gpuClusteringConstants.h"        // !

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace pixelgpudetails {

  TrackingRecHit2D PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigis const& digis_d,
                                                       SiPixelClusters const& clusters_d,
                                                       BeamSpot const& bs_d,
                                                       pixelCPEforGPU::ParamsOnGPU const* cpeParams) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2D hits_d(nHits, cpeParams, clusters_d.clusModuleStart());

    int threadsPerBlock = 128;
    int blocks = digis_d.nModules();  // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    if (blocks)  // protect from empty events
      gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0>>>(
          cpeParams, bs_d.data(), digis_d.view(), digis_d.nDigis(), clusters_d.view(), hits_d.view());
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      //Get pointers to pass to the device
      auto hitsModuleStart = clusters_d.clusModuleStart();
      auto layerStart = cpeParams->layerGeometry().layerStart;
      auto hitsLayerStart = hits_d.hitsLayerStart();
      auto iter = std::views::iota(0, 11);
      std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto i) {
        hitsLayerStart[i] = hitsModuleStart[layerStart[i]];
      });
    }

    if (nHits) {
      cms::cuda::fillManyFromVector(hits_d.phiBinner(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256);
      cudaCheck(cudaGetLastError());
    }

#ifdef GPU_DEBUG
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

    return hits_d;
  }

}  // namespace pixelgpudetails
