#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <cstdint>
#include <cstdio>

#include "Geometry/phase1PixelTopology.h"
#include "KokkosCore/hintLightWeight.h"
#include "KokkosCore/HistoContainer.h"
#include "KokkosDataFormats/gpuClusteringConstants.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuClustering {

#ifdef GPU_DEBUG
    __device__ uint32_t gMaxHit = 0;
#endif

    KOKKOS_INLINE_FUNCTION void countModules(Kokkos::View<uint16_t const*, KokkosExecSpace> id,
                                             Kokkos::View<uint32_t*, KokkosExecSpace> moduleStart,
                                             Kokkos::View<int32_t*, KokkosExecSpace> clusterId,
                                             int numElements,
                                             const size_t index) {
      clusterId[index] = index;
      if (::gpuClustering::InvId == id[index])
        return;
      int j = index - 1;
      while (j >= 0 and id[j] == ::gpuClustering::InvId)
        --j;
      if (j < 0 or id[j] != id[index]) {
        // boundary... replacing atomicInc with explicit logic
        auto loc = Kokkos::atomic_fetch_add(&moduleStart(0), 1);
        assert(moduleStart(0) < ::gpuClustering::MaxNumModules);
        moduleStart(loc + 1) = index;
      }
    }
  }  // namespace gpuClustering
}  // namespace KOKKOS_NAMESPACE

namespace gpuClustering {
  //  __launch_bounds__(256,4)
  template <typename ExecSpace>
  void findClus(Kokkos::View<const uint16_t*, ExecSpace> id,           // module id of each pixel
                Kokkos::View<const uint16_t*, ExecSpace> x,            // local coordinates of each pixel
                Kokkos::View<const uint16_t*, ExecSpace> y,            //
                Kokkos::View<const uint32_t*, ExecSpace> moduleStart,  // index of the first pixel of each module
                Kokkos::View<uint32_t*, ExecSpace> nClustersInModule,  // output: number of clusters found in each module
                Kokkos::View<uint32_t*, ExecSpace> moduleId,           // output: module id of each module
                Kokkos::View<int*, ExecSpace> clusterId,               // output: cluster id of each pixel
                int numElements,
                Kokkos::TeamPolicy<ExecSpace>& teamPolicy,
                ExecSpace const& execSpace) {
    using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    using shared_team_view = Kokkos::View<uint32_t, typename ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    size_t shared_view_bytes = shared_team_view::shmem_size();

    constexpr int maxPixInModule = 4000;
    constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
    using Hist = cms::kokkos::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;

    Kokkos::View<Hist*, ExecSpace> d_hist(Kokkos::ViewAllocateWithoutInitializing("d_hist"), teamPolicy.league_size());
    Kokkos::View<int*, ExecSpace> d_msize(Kokkos::ViewAllocateWithoutInitializing("d_msize"), teamPolicy.league_size());

    int loop_count = Hist::totbins();
    Kokkos::parallel_for(
        "init_hist_off", hintLightWeight(teamPolicy), KOKKOS_LAMBDA(const member_type& teamMember) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, loop_count),
                               [&](const int index) { d_hist(teamMember.league_rank()).off[index] = 0; });
        });

    Kokkos::parallel_for(
        "findClus_msize", hintLightWeight(teamPolicy), KOKKOS_LAMBDA(const member_type& teamMember) {
          if (teamMember.league_rank() >= static_cast<int>(moduleStart(0)))
            return;

          int firstPixel = moduleStart(1 + teamMember.league_rank());
          auto thisModuleId = id(firstPixel);
          assert(thisModuleId < ::gpuClustering::MaxNumModules);

          auto first = firstPixel + teamMember.team_rank();

          // find the index of the first pixel not belonging to this module (or invalid)
          d_msize(teamMember.league_rank()) = numElements;
          teamMember.team_barrier();

          // skip threads not associated to an existing pixel
          for (int i = first; i < numElements; i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            if (id(i) != thisModuleId) {  // find the first pixel in a different module
              Kokkos::atomic_fetch_min(&d_msize(teamMember.league_rank()), i);
              break;
            }
          }

          assert((d_msize(teamMember.league_rank()) == numElements) or
                 ((d_msize(teamMember.league_rank()) < numElements) and
                  (id(d_msize(teamMember.league_rank())) != thisModuleId)));

          // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
          if (0 == teamMember.team_rank()) {
            if (d_msize(teamMember.league_rank()) - firstPixel > maxPixInModule) {
              printf("too many pixels in module %d: %d > %d\n",
                     thisModuleId,
                     d_msize(teamMember.league_rank()) - firstPixel,
                     maxPixInModule);
              d_msize(teamMember.league_rank()) = maxPixInModule + firstPixel;
            }
          }

          teamMember.team_barrier();
          assert(d_msize(teamMember.league_rank()) - firstPixel <= maxPixInModule);

          // fill histo
          for (int i = first; i < d_msize(teamMember.league_rank()); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            d_hist(teamMember.league_rank()).count(y(i));
          }
        });

    Hist::finalize(d_hist, teamPolicy.league_size(), execSpace);
    int shared_view_level = 0;
    Kokkos::parallel_for(
        "findClus_msize",
        hintLightWeight(teamPolicy.set_scratch_size(shared_view_level, Kokkos::PerTeam(shared_view_bytes))),
        KOKKOS_LAMBDA(const member_type& teamMember) {
          if (uint32_t(teamMember.league_rank()) >= moduleStart(0))
            return;

          auto firstPixel = moduleStart(1 + teamMember.league_rank());
          auto first = firstPixel + teamMember.team_rank();

          for (int i = first; i < d_msize(teamMember.league_rank()); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            d_hist(teamMember.league_rank()).fill(y(i), i - firstPixel);
          }

          const uint32_t hist_size = d_hist(teamMember.league_rank()).size();

#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
          const uint32_t maxiter = hist_size;
#else
          const uint32_t maxiter = 16;
#endif

          constexpr int maxNeighbours = 10;
          assert((hist_size / teamMember.team_size()) <= maxiter);
      // nearest neighbour

#if defined KOKKOS_BACKEND_CUDA || defined KOKKOS_BACKEND_HIP
          uint16_t nn[maxiter][maxNeighbours];
          uint8_t nnn[maxiter];

          for (uint32_t k = 0; k < maxiter; ++k) {
            nnn[k] = 0;
            for (uint32_t l = 0; l < maxNeighbours; ++l)
              nn[k][l] = 0;
          }
#else

          uint16_t** nn = new uint16_t*[maxiter];
          uint8_t* nnn = new uint8_t[maxiter];
          for (uint32_t k = 0; k < maxiter; ++k) {
            nnn[k] = 0;
            nn[k] = new uint16_t[maxNeighbours];
            // cuda version does not iniitalize nn
            for (uint32_t l = 0; l < maxNeighbours; ++l)
              nn[k][l] = 0;
          }
#endif

          teamMember.team_barrier();  // for hit filling!

          // fill NN
          auto thisModuleId = id(firstPixel);
          for (uint32_t j = teamMember.team_rank(), k = 0U; j < hist_size; j += teamMember.team_size(), ++k) {
            assert(k < maxiter);
            auto p = d_hist(teamMember.league_rank()).begin() + j;
            auto i = *p + firstPixel;
            assert(id(i) != ::gpuClustering::InvId);
            assert(id(i) == thisModuleId);  // same module
            int be = Hist::bin(y(i) + 1);
            auto e = d_hist(teamMember.league_rank()).end(be);
            ++p;
            assert(0 == nnn[k]);
            for (; p < e; ++p) {
              auto m = (*p) + firstPixel;
              assert(m != i);
              assert(int(y(m)) - int(y(i)) >= 0);
              assert(int(y(m)) - int(y(i)) <= 1);
              if (std::abs(int(x(m)) - int(x(i))) > 1)
                continue;
              auto l = nnn[k]++;
              assert(l < maxNeighbours);
              nn[k][l] = *p;
            }
          }

          // for each pixel, look at all the pixels until the end of the module;
          // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
          // after the loop, all the pixel in each cluster should have the id equeal to the lowest
          // pixel in the cluster ( clus[i] == i ).
          int more = 1;
          int nloops = 0;
          while (more) {
            if (1 == nloops % 2) {
              for (uint16_t j = teamMember.team_rank(), k = 0U; j < d_hist(teamMember.league_rank()).size();
                   j += teamMember.team_size(), ++k) {
                auto p = d_hist(teamMember.league_rank()).begin() + j;
                auto i = *p + firstPixel;
                auto m = clusterId(i);
                while (m != clusterId(m)) {
                  m = clusterId(m);
                }
                clusterId(i) = m;
              }
            } else {
              more = 0;
              for (uint16_t j = teamMember.team_rank(), k = 0U; j < d_hist(teamMember.league_rank()).size();
                   j += teamMember.team_size(), ++k) {
                auto p = d_hist(teamMember.league_rank()).begin() + j;
                auto i = *p + firstPixel;
                for (uint16_t kk = 0; kk < nnn[k]; ++kk) {
                  auto l = nn[k][kk];
                  auto m = l + firstPixel;
                  assert(m != i);
                  auto old = Kokkos::atomic_fetch_min(&clusterId(m), clusterId(i));
                  if (old != clusterId(i)) {
                    // end the loop only if no changes were applied
                    more = 1;
                  }
                  Kokkos::atomic_fetch_min(&clusterId(i), old);
                }  // nnloop
              }    // pixel loop
            }
            ++nloops;
            teamMember.team_reduce(Kokkos::Sum<decltype(more)>(more));
          }  // end while

          shared_team_view foundClusters(teamMember.team_scratch(shared_view_level));
          foundClusters() = 0;
          teamMember.team_barrier();

          // find the number of different clusters, identified by a pixels with clus[i] == i;
          // mark these pixels with a negative id.
          for (int i = first; i < d_msize(teamMember.league_rank()); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            if (clusterId(i) == i) {
              auto old = Kokkos::atomic_fetch_add(&foundClusters(), 1);
              assert(foundClusters() < 0xffffffff);
              clusterId(i) = -(old + 1);
            }
          }
          teamMember.team_barrier();

          // propagate the negative id to all the pixels in the cluster.
          for (int i = first; i < d_msize(teamMember.league_rank()); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            if (clusterId(i) >= 0) {
              // mark each pixel in a cluster with the same id as the first one
              clusterId(i) = clusterId(clusterId(i));
            }
          }
          teamMember.team_barrier();

          // adjust the cluster id to be a positive value starting from 0
          for (int i = first; i < d_msize(teamMember.league_rank()); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId) {  // skip invalid pixels
              clusterId(i) = -9999;
              continue;
            }
            clusterId(i) = -clusterId(i) - 1;
          }
          teamMember.team_barrier();

          if (teamMember.team_rank() == 0) {
            nClustersInModule(thisModuleId) = foundClusters();
            moduleId(teamMember.league_rank()) = thisModuleId;
          }
        });
  }  // end findClus()
}  // namespace gpuClustering
#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
