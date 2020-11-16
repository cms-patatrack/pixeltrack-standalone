#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksDBSCAN_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksDBSCAN_h

#include "KokkosCore/kokkos_assert.h"
#include "KokkosCore/HistoContainer.h"

#include "gpuVertexFinder.h"
#include "gpuClusterFillHist.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {

    template <typename Histo>
    KOKKOS_INLINE_FUNCTION void clusterTracksDBSCAN(
        Kokkos::View<ZVertices, KokkosExecSpace> vdata,
        Kokkos::View<WorkSpace, KokkosExecSpace> vws,
        Kokkos::View<Histo*, KokkosExecSpace> vhist,
        int minT,       // min number of neighbours to be "seed"
        float eps,      // max absolute distance to cluster
        float errmax,   // max error to be "seed"
        float chi2max,  // max normalized distance to cluster
        const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false

      const auto leagueRank = team_member.league_rank();
      const auto teamRank = team_member.team_rank();
      const auto teamSize = team_member.team_size();
      const auto id = leagueRank * teamSize + teamRank;

      auto er2mx = errmax * errmax;

      auto& __restrict__ data = *vdata.data();
      auto& __restrict__ ws = *vws.data();

      auto nt = ws.ntrks;
      float const* __restrict__ zt = ws.zt;
      float const* __restrict__ ezt2 = ws.ezt2;

      uint32_t& nvFinal = data.nvFinal;
      uint32_t& nvIntermediate = ws.nvIntermediate;

      uint8_t* __restrict__ izt = ws.izt;
      int32_t* __restrict__ nn = data.ndof;
      int32_t* __restrict__ iv = ws.iv;

      auto* localHist = &vhist(leagueRank);

      assert(localHist->size() == nt);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt),
                           [=](int i) { localHist->fill(izt[i], uint16_t(i)); });
      team_member.team_barrier();

      // count neighbours
      // TODO: can't use parallel_for+TeamThreadRange because of "continue"
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        if (ezt2[i] > er2mx)
          continue;
        auto loop = [&](uint32_t j) {
          if (i == j)
            return;
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > eps)
            return;
          //        if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
          nn[i]++;
        };

        forEachInBins(localHist, izt[i], 1, loop);
      }

      team_member.team_barrier();

      // find NN with smaller z...
      // TODO: can't use parallel_for+TeamThreadRange because of "continue"
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        if (nn[i] < minT)
          continue;  // DBSCAN core rule
        float mz = zt[i];
        auto loop = [&](uint32_t j) {
          if (zt[j] >= mz)
            return;
          if (nn[j] < minT)
            return;  // DBSCAN core rule
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > eps)
            return;
          //        if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
          mz = zt[j];
          iv[i] = j;  // assign to cluster (better be unique??)
        };
        forEachInBins(localHist, izt[i], 1, loop);
      }

      team_member.team_barrier();

#ifdef GPU_DEBUG
      //  mini verification
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
        if (iv[i] != int(i))
          assert(iv[iv[i]] != int(i));
      });
      team_member.team_barrier();
#endif

      // consolidate graph (percolate index of seed)
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
        auto m = iv[i];
        while (m != iv[m])
          m = iv[m];
        iv[i] = m;
      });

      team_member.team_barrier();

#ifdef GPU_DEBUG
      //  mini verification
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
        if (iv[i] != int(i))
          assert(iv[iv[i]] != int(i));
      });
      team_member.team_barrier();
#endif

#ifdef GPU_DEBUG
      // and verify that we did not spit any cluster...
      // TODO: can't use parallel_for+TeamThreadRange because of "continue"
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        if (nn[i] < minT)
          continue;  // DBSCAN core rule
        assert(zt[iv[i]] <= zt[i]);
        auto loop = [&](uint32_t j) {
          if (nn[j] < minT)
            return;  // DBSCAN core rule
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > eps)
            return;
          //  if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
          // they should belong to the same cluster, isn't it?
          if (iv[i] != iv[j]) {
            printf("ERROR %d %d %f %f %d\n", i, iv[i], zt[i], zt[iv[i]], iv[iv[i]]);
            printf("      %d %d %f %f %d\n", j, iv[j], zt[j], zt[iv[j]], iv[iv[j]]);
            ;
          }
          assert(iv[i] == iv[j]);
        };
        forEachInBins(localHist, izt[i], 1, loop);
      }
      team_member.team_barrier();
#endif

      // collect edges (assign to closest cluster of closest point??? here to closest point)
      // TODO: can't use parallel_for+TeamThreadRange because of "continue"
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        //    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
        if (nn[i] >= minT)
          continue;  // DBSCAN edge rule
        float mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < minT)
            return;  // DBSCAN core rule
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;  // needed?
          mdist = dist;
          iv[i] = iv[j];  // assign to cluster (better be unique??)
        };
        forEachInBins(localHist, izt[i], 1, loop);
      }

      unsigned int* foundClusters =
          static_cast<unsigned int*>(team_member.team_shmem().get_shmem(sizeof(unsigned int)));
      foundClusters[0] = 0;
      team_member.team_barrier();

      // find the number of different clusters, identified by a tracks with clus[i] == i;
      // mark these tracks with a negative id.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
        if (iv[i] == int(i)) {
          if (nn[i] >= minT) {
            auto old = Kokkos::atomic_fetch_add(foundClusters, 1);
            iv[i] = -(old + 1);
          } else {  // noise
            iv[i] = -9998;
          }
        }
      });
      team_member.team_barrier();

      assert(foundClusters[0] < ZVertices::MAXVTX);

      // propagate the negative id to all the tracks in the cluster.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) {
        if (iv[i] >= 0) {
          // mark each track in a cluster with the same id as the first one
          iv[i] = iv[iv[i]];
        }
      });
      team_member.team_barrier();

      // adjust the cluster id to be a positive value starting from 0
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [=](int i) { iv[i] = -iv[i] - 1; });

      nvIntermediate = nvFinal = foundClusters[0];

      if (verbose && 0 == id)
        printf("found %d proto vertices\n", foundClusters[0]);
    }

    template <typename ExecSpace>
    void clusterTracksDBSCANHost(Kokkos::View<ZVertices, ExecSpace> vdata,
                                 Kokkos::View<WorkSpace, ExecSpace> vws,
                                 int minT,       // min number of neighbours to be "seed"
                                 float eps,      // max absolute distance to cluster
                                 float errmax,   // max error to be "seed"
                                 float chi2max,  // max normalized distance to cluster
                                 const ExecSpace& execSpace,
                                 const Kokkos::TeamPolicy<ExecSpace>& policy) {
      using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

      auto leagueSize = policy.league_size();

      using Hist = cms::kokkos::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;
      Kokkos::View<Hist*, ExecSpace> vhist(Kokkos::ViewAllocateWithoutInitializing("vhist"), leagueSize);

      Kokkos::parallel_for(
          "clusterFillHist", policy, KOKKOS_LAMBDA(const member_type& team_member) {
            clusterFillHist(vdata, vws, vhist, minT, eps, errmax, chi2max, team_member);
          });

      Hist::finalize(vhist, leagueSize, execSpace);

      Kokkos::parallel_for(
          "clusterTracksDBSCAN", policy, KOKKOS_LAMBDA(const member_type& team_member) {
            clusterTracksDBSCAN(vdata, vws, vhist, minT, eps, errmax, chi2max, team_member);
          });
    }
  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksDBSCAN_h
