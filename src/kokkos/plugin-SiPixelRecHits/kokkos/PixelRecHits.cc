// C++ headers
#include <algorithm>
#include <numeric>

// CMSSW headers
#ifdef TODO
#include "plugin-SiPixelClusterizer/kokkos/SiPixelRawToClusterGPUKernel.h"  // !
#include "plugin-SiPixelClusterizer/gpuClusteringConstants.h"               // !
#endif

#include "CondFormats/pixelCPEforGPU.h"
//#include "KokkosCore/kokkos_assert.h"

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

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
        TeamPolicy policy(execSpace, digis_d.nModules(), Kokkos::AUTO());
        Kokkos::parallel_for(
            "getHits",
            policy.set_scratch_size(0, Kokkos::PerTeam(sizeof(pixelCPEforGPU::ClusParams))),
            KOKKOS_LAMBDA(MemberType const& teamMember) {
              gpuPixelRecHits::getHits(
                  cpeParams.data(), bs_d.data(), digisView, nDigis, clustersView, hitsView, teamMember);
            });
      }

#ifdef GPU_DEBUG
      execSpace.fence();
#endif

      // assuming full warp of threads is better than a smaller number...
      if (nHits) {
        auto clusModuleStart = clusters_d.clusModuleStart();
        auto hitsLayerStart = hits_d.hitsLayerStart();
        Kokkos::parallel_for(
            "hitsLayerStart", Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, 11), KOKKOS_LAMBDA(const size_t i) {
              // TODO: for some reason uncommenting the assert leads to
              // cudaFuncGetAttributes( &attr, cuda_parallel_launch_local_memory<DriverType>) error( cudaErrorInvalidDeviceFunction): invalid device function .../pixeltrack-standalone/external/kokkos/install/include/Cuda/Kokkos_Cuda_KernelLaunch.hpp:448
              //assert(0 == clusModuleStart[0]);

              hitsLayerStart[i] = clusModuleStart[cpeParams().layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
              printf("LayerStart %d %d: %d\n", i, cpeParams().layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
            });
      }

      if (nHits) {
        cms::kokkos::fillManyFromVector(
            hits_d.phiBinner(), 10, hits_d.c_iphi(), hits_d.c_hitsLayerStart(), nHits, 256, execSpace);
      }

#ifdef GPU_DEBUG
      execSpace.fence();
#endif
      return hits_d;
    }

  }  // namespace pixelgpudetails
}  // namespace KOKKOS_NAMESPACE
