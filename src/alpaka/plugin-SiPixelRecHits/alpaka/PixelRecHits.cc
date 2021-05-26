#include "CondFormats/pixelCPEforGPU.h"

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace {
  struct setHitsLayerStart {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                  uint32_t const* __restrict__ hitsModuleStart,
                                  pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                  uint32_t* hitsLayerStart) const {
      assert(0 == hitsModuleStart[0]);

      cms::alpakatools::for_each_element_in_grid(acc, 11, [&](uint32_t i) {
        hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
        printf("LayerStart %d %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
      });
    }
  };
}  // namespace

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace pixelgpudetails {

    TrackingRecHit2DAlpaka PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisAlpaka const& digis_d,
                                                               SiPixelClustersAlpaka const& clusters_d,
                                                               BeamSpotAlpaka const& bs_d,
                                                               pixelCPEforGPU::ParamsOnGPU const* cpeParams) const {
      auto nHits = clusters_d.nClusters();
      TrackingRecHit2DAlpaka hits_d(nHits, cpeParams, clusters_d.clusModuleStart());

      const int threadsPerBlockOrElementsPerThread = 128;
      const int blocks = digis_d.nModules();  // active modules (with digis)
      const WorkDiv1& getHitsWorkDiv =
          cms::alpakatools::make_workdiv(Vec1::all(blocks), Vec1::all(threadsPerBlockOrElementsPerThread));

#ifdef GPU_DEBUG
      std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif

      Queue queue(device);

      if (blocks) {  // protect from empty events
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(getHitsWorkDiv,
                                                       gpuPixelRecHits::getHits(),
                                                       cpeParams,
                                                       bs_d.data(),
                                                       digis_d.view(),
                                                       digis_d.nDigis(),
                                                       clusters_d.view(),
                                                       hits_d.view()));
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
#endif

      // assuming full warp of threads is better than a smaller number...
      if (nHits) {
        const WorkDiv1& oneBlockWorkDiv = cms::alpakatools::make_workdiv(Vec1::all(1u), Vec1::all(32u));
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc1>(
                oneBlockWorkDiv, setHitsLayerStart(), clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart()));
      }

      if (nHits) {
        cms::alpakatools::fillManyFromVector(
            hits_d.phiBinner(), 10, hits_d.c_iphi(), hits_d.c_hitsLayerStart(), nHits, 256, queue);
      }

      //#ifdef GPU_DEBUG
      //alpaka::wait(queue);
      //#endif

      alpaka::wait(queue);
      return hits_d;
    }

  }  // namespace pixelgpudetails

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
