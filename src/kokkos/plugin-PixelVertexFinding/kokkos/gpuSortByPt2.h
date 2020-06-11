#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h

#ifdef TODO
#ifdef __CUDA_ARCH__
#include "CUDACore/radixSort.h"
#endif
#endif  // TODO

#include "gpuVertexFinder.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {

    KOKKOS_INLINE_FUNCTION void sortByPt2(Kokkos::View<ZVertices*, KokkosExecSpace> vdata,
                                          Kokkos::View<WorkSpace*, KokkosExecSpace> vws,
                                          const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      auto& __restrict__ data = *vdata.data();
      auto& __restrict__ ws = *vws.data();
      auto nt = ws.ntrks;
      float const* __restrict__ ptt2 = ws.ptt2;
      uint32_t const& nvFinal = data.nvFinal;

      int32_t const* __restrict__ iv = ws.iv;
      float* __restrict__ ptv2 = data.ptv2;
      uint16_t* __restrict__ sortInd = data.sortInd;

      // if (threadIdx.x == 0)
      //    printf("sorting %d vertices\n",nvFinal);

      if (nvFinal < 1)
        return;

      // fill indexing
      for (unsigned int i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        data.idv[ws.itrk[i]] = iv[i];
      }

      // can be done asynchronoisly at the end of previous event
      for (unsigned int i = team_member.team_rank(); i < nvFinal; i += team_member.team_size()) {
        ptv2[i] = 0;
      }
      team_member.team_barrier();

      for (unsigned int i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        if (iv[i] > 9990)
          continue;
        Kokkos::atomic_add(&ptv2[iv[i]], ptt2[i]);
      }

      team_member.team_barrier();

      if (1 == nvFinal) {
        if (team_member.team_rank() == 0)
          sortInd[0] = 0;
        return;
      }
#ifdef __CUDA_ARCH__
#ifdef TODO
      __shared__ uint16_t sws[1024];
      // sort using only 16 bits
      radixSort<float, 2>(ptv2, sortInd, sws, nvFinal);
#endif  // TODO
#else
      for (uint16_t i = 0; i < nvFinal; ++i)
        sortInd[i] = i;
      std::sort(sortInd, sortInd + nvFinal, [&](auto i, auto j) { return ptv2[i] < ptv2[j]; });
#endif
    }

    KOKKOS_INLINE_FUNCTION void sortByPt2Kernel(Kokkos::View<ZVertices*, KokkosExecSpace> vdata,
                                                Kokkos::View<WorkSpace*, KokkosExecSpace> vws,
                                                const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      sortByPt2(vdata, vws, team_member);
    }

  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
