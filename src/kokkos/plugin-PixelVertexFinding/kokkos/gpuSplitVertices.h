#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h

#ifdef TODO
#include "CUDACore/cuda_assert.h"
#endif  // TODO

#include "gpuVertexFinder.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {

    KOKKOS_INLINE_FUNCTION void splitVertices(Kokkos::View<ZVertices, KokkosExecSpace> vdata,
                                              Kokkos::View<WorkSpace, KokkosExecSpace> vws,
                                              float maxChi2,
                                              const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false

      auto& __restrict__ data = *vdata.data();
      auto& __restrict__ ws = *vws.data();
      auto nt = ws.ntrks;
      float const* __restrict__ zt = ws.zt;
      float const* __restrict__ ezt2 = ws.ezt2;
      float* __restrict__ zv = data.zv;
      float* __restrict__ wv = data.wv;
      float const* __restrict__ chi2 = data.chi2;
      uint32_t& nvFinal = data.nvFinal;

      int32_t const* __restrict__ nn = data.ndof;
      int32_t* __restrict__ iv = ws.iv;

#ifdef TODO
      assert(vdata.data());
      assert(zt);
#endif

      // one vertex per block
      for (unsigned kv = team_member.league_rank(); kv < nvFinal; kv += team_member.league_size()) {
        if (nn[kv] < 4)
          continue;
        if (chi2[kv] < maxChi2 * float(nn[kv]))
          continue;

        constexpr int MAXTK = 512;
#ifdef TODO
        assert(nn[kv] < MAXTK);
#endif
        if (nn[kv] >= MAXTK)
          continue;                                                                              // too bad FIXME
        uint32_t* it = (uint32_t*)team_member.team_shmem().get_shmem(sizeof(uint32_t) * MAXTK);  // track index
        float* zz = (float*)team_member.team_shmem().get_shmem(sizeof(float) * MAXTK);           // z pos
        uint8_t* newV = (uint8_t*)team_member.team_shmem().get_shmem(sizeof(uint8_t) * MAXTK);   // 0 or 1
        float* ww = (float*)team_member.team_shmem().get_shmem(sizeof(float) * MAXTK);           // z weight

        uint32_t* nq =
            (uint32_t*)team_member.team_shmem().get_shmem(sizeof(uint32_t));  // number of track for this vertx
        nq[0] = 0;
        team_member.team_barrier();

        // copy to local
        for (unsigned k = team_member.team_rank(); k < nt; k += team_member.team_size()) {
          if (iv[k] == int(kv)) {
            uint32_t old;
            if (nq[0] >= MAXTK) {
              old = Kokkos::atomic_exchange(nq, 0);
            } else {
              old = Kokkos::atomic_fetch_add(nq, 1);
            }
            zz[old] = zt[k] - zv[kv];
            newV[old] = zz[old] < 0 ? 0 : 1;
            ww[old] = 1.f / ezt2[k];
            it[old] = k;
          }
        }

        // the new vertices
        float* znew = (float*)team_member.team_shmem().get_shmem(sizeof(float) * 2);
        float* wnew = (float*)team_member.team_shmem().get_shmem(sizeof(float) * 2);

        team_member.team_barrier();
#ifdef TODO
        assert(int(nq[0]) == nn[kv] + 1);
#endif

        int maxiter = 20;

        // kt-min....
        bool more = true;
        bool* more_list = (bool*)team_member.team_shmem().get_shmem(sizeof(bool) * team_member.team_size());
        while (more) {
          more_list[team_member.team_rank()] = false;
          if (0 == team_member.team_rank()) {
            znew[0] = 0;
            znew[1] = 0;
            wnew[0] = 0;
            wnew[1] = 0;
          }
          team_member.team_barrier();
          for (unsigned k = team_member.team_rank(); k < nq[0]; k += team_member.team_size()) {
            auto i = newV[k];
            Kokkos::atomic_add(&znew[i], zz[k] * ww[k]);
            Kokkos::atomic_add(&wnew[i], ww[k]);
          }
          team_member.team_barrier();
          if (0 == team_member.team_rank()) {
            znew[0] /= wnew[0];
            znew[1] /= wnew[1];
          }
          team_member.team_barrier();
          for (unsigned k = team_member.team_rank(); k < nq[0]; k += team_member.team_size()) {
            auto d0 = fabs(zz[k] - znew[0]);
            auto d1 = fabs(zz[k] - znew[1]);
            auto newer = d0 < d1 ? 0 : 1;
            more_list[team_member.team_rank()] |= newer != newV[k];
            newV[k] = newer;
          }
          --maxiter;
          if (maxiter <= 0)
            more_list[team_member.team_rank()] = false;
          team_member.team_barrier();

          int sum = 0;
          for (int i = 0; i < team_member.team_size(); i++)
            sum += int(more_list[i]);
          if (sum == 0)
            more = false;
        }

        // avoid empty vertices
        if (0 == wnew[0] || 0 == wnew[1])
          continue;

        // quality cut
        auto dist2 = (znew[0] - znew[1]) * (znew[0] - znew[1]);

        auto chi2Dist = dist2 / (1.f / wnew[0] + 1.f / wnew[1]);

        if (verbose && 0 == team_member.team_rank())
          printf("inter %d %f %f\n", 20 - maxiter, chi2Dist, dist2 * wv[kv]);

        if (chi2Dist < 4)
          continue;

        // get a new global vertex
        uint32_t* igv = (uint32_t*)team_member.team_shmem().get_shmem(sizeof(uint32_t));

        if (0 == team_member.team_rank())
          igv[0] = Kokkos::atomic_fetch_add(&ws.nvIntermediate, 1);
        team_member.team_barrier();
        for (unsigned k = team_member.team_rank(); k < nq[0]; k += team_member.team_size()) {
          if (1 == newV[k])
            iv[it[k]] = igv[0];
        }

      }  // loop on vertices
    }

    KOKKOS_INLINE_FUNCTION void splitVerticesKernel(
        Kokkos::View<ZVertices, KokkosExecSpace> vdata,
        Kokkos::View<WorkSpace, KokkosExecSpace> vws,
        float maxChi2,
        const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      splitVertices(vdata, vws, maxChi2, team_member);
    }

  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
