// C++ headers
#include <algorithm>
#include <numeric>

// CMSSW headers
#ifdef TODO
#include "plugin-SiPixelClusterizer/kokkos/SiPixelRawToClusterGPUKernel.h"  // !
#include "plugin-SiPixelClusterizer/gpuClusteringConstants.h"               // !
#endif

#include "CondFormats/pixelCPEforGPU.h"

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

#ifdef TODO
namespace {
  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                    uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    assert(0 == hitsModuleStart[0]);

    if (i < 11) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace
#endif

namespace KOKKOS_NAMESPACE {

  namespace pixelgpudetails {

    TrackingRecHit2DKokkos<KokkosExecSpace> PixelRecHitGPUKernel::makeHitsAsync(
        SiPixelDigisKokkos<KokkosExecSpace> const& digis_d,
        SiPixelClustersKokkos<KokkosExecSpace> const& clusters_d,
        BeamSpotKokkos<KokkosExecSpace> const& bs_d,
        Kokkos::View<pixelCPEforGPU::ParamsOnGPU const, KokkosExecSpace> const& cpeParams,
        KokkosExecSpace const& execSpace) const {
      auto nHits = clusters_d.nClusters();
      TrackingRecHit2DKokkos<KokkosExecSpace> hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), execSpace);

      auto nDigis = digis_d.nDigis();
      auto digisView = digis_d.view();
      auto clustersView = clusters_d.view();
      auto hitsView = hits_d.view();

      using TeamPolicy = Kokkos::TeamPolicy<KokkosExecSpace>;
      using MemberType = TeamPolicy::member_type;

#ifdef GPU_DEBUG
      std::cout << "launching getHits kernel for " << digis.nModules() << " teams" << std::endl;
#endif

      if (digis_d.nModules() > 0) {  // protect from empty events
        // one team for each active module (with digis)
        TeamPolicy policy(execSpace, digis_d.nModules(), 128);  // TODO: see if can use Kokkos::AUTO()
        Kokkos::parallel_for(
            policy.set_scratch_size(0, Kokkos::PerTeam(sizeof(pixelCPEforGPU::ClusParams))),
            KOKKOS_LAMBDA(MemberType const& teamMember) {
              gpuPixelRecHits::getHits(
                  cpeParams.data(), bs_d.data(), digisView, nDigis, clustersView, hitsView, teamMember);
            });
      }

#ifdef GPU_DEBUG
      execSpace.fence();
#endif

#ifdef TODO
      // assuming full warp of threads is better than a smaller number...
      if (nHits) {
        setHitsLayerStart<<<1, 32, 0, stream>>>(clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart());
        cudaCheck(cudaGetLastError());
      }

      if (nHits) {
        auto hws = cms::cuda::make_device_unique<uint8_t[]>(TrackingRecHit2DSOAView::Hist::wsSize(), stream);
        cms::cuda::fillManyFromVector(
            hits_d.phiBinner(), hws.get(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256, stream);
        cudaCheck(cudaGetLastError());
      }

#ifdef GPU_DEBUG
      cudaDeviceSynchronize();
      cudaCheck(cudaGetLastError());
#endif
#endif
      return hits_d;
    }

  }  // namespace pixelgpudetails
}  // namespace KOKKOS_NAMESPACE
