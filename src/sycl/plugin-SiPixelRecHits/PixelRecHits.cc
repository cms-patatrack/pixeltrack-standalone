// C++ headers
#include <algorithm>
#include <numeric>

// SYCL runtime
#include <sycl/sycl.hpp>

// CMSSW headers
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLDataFormats/gpuClusteringConstants.h"                  // !
#include "plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.h"  // !

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace {
  void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                         pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                         uint32_t* hitsLayerStart,
                         sycl::nd_item<1> item) {
    auto i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

    assert(0 == hitsModuleStart[0]);

    if (i < 11) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart[%d] %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

  TrackingRecHit2DSYCL PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisSYCL const& digis_d,
                                                           SiPixelClustersSYCL const& clusters_d,
                                                           BeamSpotSYCL const& bs_d,
                                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                           sycl::queue stream) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2DSYCL hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), stream);

    int threadsPerBlock = 128;
    int blocks = digis_d.nModules();  // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    if (blocks)  // protect from empty events
      stream.submit([&](sycl::handler& cgh) {
        auto cpeParams_kernel = cpeParams;
        auto bs_d_kernel = bs_d.data();
        auto digis_view_kernel = digis_d.view();
        auto digis_n_kernel = digis_d.nDigis();
        auto clusters_d_kernel = clusters_d.view();
        auto hits_d_kernel = hits_d.view();
        cgh.parallel_for<class getHits_Kernel>(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock),
                                               [=](sycl::nd_item<1> item) {
                                                 gpuPixelRecHits::getHits(cpeParams_kernel,
                                                                          bs_d_kernel,
                                                                          digis_view_kernel,
                                                                          digis_n_kernel,
                                                                          clusters_d_kernel,
                                                                          hits_d_kernel,
                                                                          item);
                                               });
      });

#ifdef GPU_DEBUG
    stream.wait();
#endif

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
#ifdef GPU_DEBUG
      std::cout << "launching setHitsLayerStart kernel for 32 blocks" << std::endl;
#endif
      stream.submit([&](sycl::handler& cgh) {
        auto cpeParams_kernel = cpeParams;
        auto hits_d_kernel = hits_d.hitsLayerStart();
        auto clusters_d_kernel = clusters_d.clusModuleStart();
        cgh.parallel_for<class setHitsLayerStart_Kernel>(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> item) {
          setHitsLayerStart(clusters_d_kernel, cpeParams_kernel, hits_d_kernel, item);
        });
      });
    }
    if (nHits) {
#ifdef GPU_DEBUG
      std::cout << "launching fillManyFromVector kernel" << std::endl;
#endif
      cms::sycltools::fillManyFromVector(
          hits_d.phiBinner(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256, stream);
    }

#ifdef GPU_DEBUG
    stream.wait();
#endif
#ifdef CPU_DEBUG
    stream.wait();
#endif

    return hits_d;
  }

}  // namespace pixelgpudetails
