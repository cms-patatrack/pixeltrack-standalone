#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include "KokkosCore/hintLightWeight.h"
#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/atomic.h"
#include "KokkosDataFormats/gpuClusteringConstants.h"

namespace gpuClustering {

  template <typename MemSpace, typename ExecSpace>
  void clusterChargeCut(
      const Kokkos::View<uint16_t*, MemSpace>& id,                 // module id of each pixel
      const Kokkos::View<const uint16_t*, MemSpace>& adc,          // local coordinates of each pixel
      const Kokkos::View<const uint32_t*, MemSpace>& moduleStart,  // index of the first pixel of each module
      const Kokkos::View<uint32_t*, MemSpace>& nClustersInModule,  // output: number of clusters found in each module
      const Kokkos::View<const uint32_t*, MemSpace>& moduleId,     // output: module id of each module
      const Kokkos::View<int*, MemSpace>& clusterId,               // output: cluster id of each pixel
      int numElements,
      Kokkos::TeamPolicy<ExecSpace>& teamPolicy,
      ExecSpace const& execSpace) {
    using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    using charge_view_type = Kokkos::View<int32_t*, typename ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    size_t charge_view_bytes = charge_view_type::shmem_size(::gpuClustering::MaxNumClustersPerModules);
    using ok_view_type = Kokkos::View<uint8_t*, typename ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    size_t ok_view_bytes = ok_view_type::shmem_size(::gpuClustering::MaxNumClustersPerModules);
    using newclusid_view_type =
        Kokkos::View<uint16_t*, typename ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    size_t newclusid_view_bytes = newclusid_view_type::shmem_size(::gpuClustering::MaxNumClustersPerModules);

    auto total_shared_bytes = charge_view_bytes + ok_view_bytes + newclusid_view_bytes;

    int shared_view_level = 0;
    Kokkos::parallel_for(
        "clusterChargeCut",
        hintLightWeight(teamPolicy.set_scratch_size(shared_view_level, Kokkos::PerTeam(total_shared_bytes))),
        KOKKOS_LAMBDA(const member_type& teamMember) {
          if (uint32_t(teamMember.league_rank()) >= moduleStart(0))
            return;

          auto firstPixel = moduleStart(1 + teamMember.league_rank());
          auto thisModuleId = id(firstPixel);
          assert(thisModuleId < ::gpuClustering::MaxNumModules);
          assert(thisModuleId == moduleId(teamMember.league_rank()));

          auto nclus = nClustersInModule(thisModuleId);
          if (nclus == 0)
            return;

          Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
            if (nclus > ::gpuClustering::MaxNumClustersPerModules) {
              printf("Warning too many clusters in module %d in block %d: %d > %d\n",
                     thisModuleId,
                     teamMember.league_rank(),
                     nclus,
                     ::gpuClustering::MaxNumClustersPerModules);
            }
          });

          auto first = firstPixel + teamMember.team_rank();

          if (nclus > ::gpuClustering::MaxNumClustersPerModules) {
            // remove excess  FIXME find a way to cut charge first....
            for (int i = first; i < numElements; i += teamMember.team_size()) {
              if (id(i) == ::gpuClustering::InvId)
                continue;  // not valid
              if (id(i) != thisModuleId)
                break;  // end of module
              if (clusterId(i) >= ::gpuClustering::MaxNumClustersPerModules) {
                id(i) = ::gpuClustering::InvId;
                clusterId(i) = ::gpuClustering::InvId;
              }
            }
            nclus = ::gpuClustering::MaxNumClustersPerModules;
          }

#ifdef GPU_DEBUG
          if (thisModuleId % 100 == 1) {
            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
              printf("start clusterizer for module %d in block %d\n", thisModuleId, teamMember.league_rank());
            });
          }
#endif

          charge_view_type charge(teamMember.team_scratch(shared_view_level),
                                  ::gpuClustering::MaxNumClustersPerModules);
          ok_view_type ok(teamMember.team_scratch(shared_view_level), ::gpuClustering::MaxNumClustersPerModules);
          newclusid_view_type newclusId(teamMember.team_scratch(shared_view_level),
                                        ::gpuClustering::MaxNumClustersPerModules);

          assert(nclus <= ::gpuClustering::MaxNumClustersPerModules);
          for (uint32_t i = teamMember.team_rank(); i < ::gpuClustering::MaxNumClustersPerModules;
               i += teamMember.team_size()) {
            charge(i) = 0;
            ok(i) = 0;
            newclusId(i) = 0;
          }
          teamMember.team_barrier();

          for (int i = first; i < numElements; i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)
              continue;  // not valid
            if (id(i) != thisModuleId)
              break;  // end of module
            cms::kokkos::atomic_add<int>(&charge(clusterId(i)), adc(i));
          }
          teamMember.team_barrier();

          auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
          for (uint32_t i = teamMember.team_rank(); i < nclus; i += teamMember.team_size()) {
            newclusId(i) = ok(i) = charge(i) > chargeCut ? 1 : 0;
          }
          teamMember.team_barrier();

          // renumber
          // cuda version was a blockPrefixScan()
          Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
            for (uint32_t i = 1; i < nclus; ++i) {
              newclusId(i) += newclusId(i - 1);
            }
          });
          teamMember.team_barrier();
          assert(nclus >= newclusId(nclus - 1));

          if (nclus == newclusId(nclus - 1))
            return;

          nClustersInModule(thisModuleId) = newclusId(nclus - 1);
          teamMember.team_barrier();

          // mark bad cluster again
          for (uint32_t i = teamMember.team_rank(); i < nclus; i += teamMember.team_size()) {
            if (0 == ok(i))
              newclusId(i) = ::gpuClustering::InvId + 1;
          }
          teamMember.team_barrier();

          // reassign id
          for (int i = first; i < numElements; i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)
              continue;  // not valid
            if (id(i) != thisModuleId)
              break;  // end of module
            clusterId(i) = newclusId(clusterId(i)) - 1;
            if (clusterId(i) == ::gpuClustering::InvId)
              id(i) = ::gpuClustering::InvId;
          }
        });

    //done
  }  // end clusterChargeCut()
}  // namespace gpuClustering
#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
