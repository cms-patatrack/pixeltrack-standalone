#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h

#include "KokkosCore/kokkos_assert.h"

#include "gpuVertexFinder.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {

    KOKKOS_INLINE_FUNCTION void fitVertices(Kokkos::View<ZVertices, KokkosExecSpace> vdata,
                                            Kokkos::View<WorkSpace, KokkosExecSpace> vws,
                                            float chi2Max,  // for outlier rejection
                                            const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false

      auto& __restrict__ data = *vdata.data();
      auto& __restrict__ ws = *vws.data();
      auto nt = ws.ntrks;
      float const* __restrict__ zt = ws.zt;
      float const* __restrict__ ezt2 = ws.ezt2;
      float* __restrict__ zv = data.zv;
      float* __restrict__ wv = data.wv;
      float* __restrict__ chi2 = data.chi2;
      uint32_t& nvFinal = data.nvFinal;
      uint32_t& nvIntermediate = ws.nvIntermediate;

      int32_t* __restrict__ nn = data.ndof;
      int32_t* __restrict__ iv = ws.iv;

      const auto teamRank = team_member.team_rank();
      const auto teamSize = team_member.team_size();
      const auto id = team_member.league_rank() * teamSize + team_member.teamRank;

      assert(vdata.data());
      assert(zt);

      assert(nvFinal <= nvIntermediate);
      nvFinal = nvIntermediate;
      auto foundClusters = nvFinal;

      // zero
      for (unsigned i = teamRank; i < foundClusters; i += teamSize) {
        zv[i] = 0;
        wv[i] = 0;
        chi2[i] = 0;
      }

      // only for test
      int* noise = (int*)team_member.team_shmem().get_shmem(sizeof(int));
      if (verbose && 0 == id)
        noise[0] = 0;

      team_member.team_barrier();

      // compute cluster location
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        if (iv[i] > 9990) {
          if (verbose)
            Kokkos::atomic_add(noise, 1);
          continue;
        }
        assert(iv[i] >= 0);
        assert(iv[i] < int(foundClusters));
        auto w = 1.f / ezt2[i];
        Kokkos::atomic_add(&zv[iv[i]], zt[i] * w);
        Kokkos::atomic_add(&wv[iv[i]], w);
      }

      team_member.team_barrier();
      // reuse nn
      for (unsigned i = teamRank; i < foundClusters; i += teamSize) {
        assert(wv[i] > 0.f);
        zv[i] /= wv[i];
        nn[i] = -1;  // ndof
      }
      team_member.team_barrier();

      // compute chi2
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        if (iv[i] > 9990)
          continue;

        auto c2 = zv[iv[i]] - zt[i];
        c2 *= c2 / ezt2[i];
        if (c2 > chi2Max) {
          iv[i] = 9999;
          continue;
        }
        Kokkos::atomic_add(&chi2[iv[i]], c2);
        Kokkos::atomic_add(&nn[iv[i]], 1);
      }
      team_member.team_barrier();
      for (unsigned i = teamRank; i < foundClusters; i += teamSize)
        if (nn[i] > 0)
          wv[i] *= float(nn[i]) / chi2[i];

      if (verbose && 0 == id)
        printf("found %d proto clusters ", foundClusters);
      if (verbose && 0 == id)
        printf("and %d noise\n", noise[0]);
    }

    KOKKOS_INLINE_FUNCTION void fitVerticesKernel(Kokkos::View<ZVertices, KokkosExecSpace> vdata,
                                                  Kokkos::View<WorkSpace, KokkosExecSpace> vws,
                                                  float chi2Max,  // for outlier rejection
                                                  const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      fitVertices(vdata, vws, chi2Max, team_member);
    }

  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
