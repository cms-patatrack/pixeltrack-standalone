#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h

#include "KokkosCore/kokkos_assert.h"
#include "KokkosCore/memoryTraits.h"
#include "KokkosCore/atomic.h"

#include "gpuVertexFinder.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {

    KOKKOS_FORCEINLINE_FUNCTION void fitVertices(const Kokkos::View<ZVertices, KokkosExecSpace, Restrict>& vdata,
                                                 const Kokkos::View<WorkSpace, KokkosExecSpace, Restrict>& vws,
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
      const auto id = team_member.league_rank() * teamSize + teamRank;

      assert(vdata.data());
      assert(zt);

      assert(nvFinal <= nvIntermediate);
      nvFinal = nvIntermediate;
      auto foundClusters = nvFinal;

      // zero
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, foundClusters), [=](int i) {
        zv[i] = 0;
        wv[i] = 0;
        chi2[i] = 0;
      });

      // only for test
      int* noise = (int*)team_member.team_shmem().get_shmem(sizeof(int));
      if (verbose && 0 == id)
        noise[0] = 0;

      team_member.team_barrier();

      // compute cluster location
      // TODO: no parallel_for + TeamThreadRange because of "continue"
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        if (iv[i] > 9990) {
          if (verbose)
            cms::kokkos::atomic_add(noise, 1);
          continue;
        }
        assert(iv[i] >= 0);
        assert(iv[i] < int(foundClusters));
        auto w = 1.f / ezt2[i];
        cms::kokkos::atomic_add(&zv[iv[i]], zt[i] * w);
        cms::kokkos::atomic_add(&wv[iv[i]], w);
      }

      team_member.team_barrier();
      // reuse nn
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, foundClusters), [=](int i) {
        assert(wv[i] > 0.f);
        zv[i] /= wv[i];
        nn[i] = -1;  // ndof
      });
      team_member.team_barrier();

      // compute chi2
      // TODO: no parallel_for + TeamThreadRange because of "continue"
      for (unsigned i = teamRank; i < nt; i += teamSize) {
        if (iv[i] > 9990)
          continue;

        auto c2 = zv[iv[i]] - zt[i];
        c2 *= c2 / ezt2[i];
        if (c2 > chi2Max) {
          iv[i] = 9999;
          continue;
        }
        cms::kokkos::atomic_add(&chi2[iv[i]], c2);
        cms::kokkos::atomic_add(&nn[iv[i]], 1);
      }
      team_member.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, foundClusters), [=](int i) {
        if (nn[i] > 0)
          wv[i] *= float(nn[i]) / chi2[i];
      });

      if (verbose && 0 == id)
        printf("found %d proto clusters ", foundClusters);
      if (verbose && 0 == id)
        printf("and %d noise\n", noise[0]);
    }

    KOKKOS_FORCEINLINE_FUNCTION void fitVerticesKernel(
        const Kokkos::View<ZVertices, KokkosExecSpace, Restrict>& vdata,
        const Kokkos::View<WorkSpace, KokkosExecSpace, Restrict>& vws,
        float chi2Max,  // for outlier rejection
        const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      fitVertices(vdata, vws, chi2Max, team_member);
    }

  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
