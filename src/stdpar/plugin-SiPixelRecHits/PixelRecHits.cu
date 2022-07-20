// C++ headers
#include <algorithm>
#include <numeric>
#include <execution>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "Framework/CountingIterator.h"
#include "plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.h"  // !
#include "plugin-SiPixelClusterizer/gpuClusteringConstants.h"        // !

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace pixelgpudetails {

  TrackingRecHit2DCUDA PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                           SiPixelClustersCUDA const& clusters_d,
                                                           BeamSpotCUDA const& bs_d,
                                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                           cudaStream_t stream) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2DCUDA hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), stream);

    int threadsPerBlock = 128;
    int blocks = digis_d.nModules();  // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    if (blocks)  // protect from empty events
      gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream>>>(
          cpeParams, bs_d.data(), digis_d.view(), digis_d.nDigis(), clusters_d.view(), hits_d.view());
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      auto hitsModuleStart = clusters_d.clusModuleStart();
      auto layerStart = cpeParams->layerGeometry().layerStart;
      auto hitsLayerStart = hits_d.hitsLayerStart();

      std::for_each_n(std::execution::par, counting_iterator{0}, 11, [=](auto i) {
        hitsLayerStart[i] = hitsModuleStart[layerStart[i]];
      });
    }

    if (nHits) {
      cms::cuda::fillManyFromVector(hits_d.phiBinner(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256, stream);
      cudaCheck(cudaGetLastError());
    }

#ifdef GPU_DEBUG
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

    return hits_d;
  }

}  // namespace pixelgpudetails
