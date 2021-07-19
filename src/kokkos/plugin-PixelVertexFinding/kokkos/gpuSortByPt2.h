#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h

#include "KokkosCore/hintLightWeight.h"

#include "gpuVertexFinder.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {

    KOKKOS_FORCEINLINE_FUNCTION void sortByPt2(const Kokkos::View<ZVertices, KokkosExecSpace, Restrict>& vdata,
                                               const Kokkos::View<WorkSpace, KokkosExecSpace, Restrict>& vws,
                                               const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      auto& __restrict__ data = *vdata.data();
      auto& __restrict__ ws = *vws.data();
      auto nt = ws.ntrks;
      float const* __restrict__ ptt2 = ws.ptt2;
      uint32_t const& nvFinal = data.nvFinal;

      int32_t const* __restrict__ iv = ws.iv;
      float* __restrict__ ptv2 = data.ptv2;
      uint16_t* __restrict__ sortInd = data.sortInd;

      const auto teamRank = team_member.team_rank();
      const auto teamSize = team_member.team_size();

      // if (threadIdx.x == 0)
      //    printf("sorting %d vertices\n",nvFinal);

      if (nvFinal < 1)
        return;

      // fill indexing
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nt), [&](int i) { data.idv[ws.itrk[i]] = iv[i]; });

      // can be done asynchronoisly at the end of previous event
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nvFinal), [=](int i) { ptv2[i] = 0; });
      team_member.team_barrier();

      // TODO: no parallel_for + TeamThreadRange because of "continue"
      for (unsigned int i = teamRank; i < nt; i += teamSize) {
        if (iv[i] > 9990)
          continue;
        Kokkos::atomic_add(&ptv2[iv[i]], ptt2[i]);
      }

      team_member.team_barrier();

      if (1 == nvFinal) {
        if (teamRank == 0)
          sortInd[0] = 0;
        return;
      }
    }

    KOKKOS_INLINE_FUNCTION void sortByPt2Kernel(const Kokkos::View<ZVertices, KokkosExecSpace, Restrict>& vdata,
                                                const Kokkos::View<WorkSpace, KokkosExecSpace, Restrict>& vws,
                                                const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
      Kokkos::abort("sortByPt2Kernel: device sort kernel not supported in Kokkos (see sortByPt2Host)");
    }

    // equivalent to CUDA sortByPt2Kernel + deep copy to host
    template <typename ExecSpace>
    void sortByPt2Host(const Kokkos::View<ZVertices, ExecSpace, Restrict>& vdata,
                       const Kokkos::View<WorkSpace, ExecSpace, Restrict>& vws,
                       typename Kokkos::View<ZVertices, ExecSpace>::HostMirror hdata,
                       const ExecSpace& execSpace,
                       const Kokkos::TeamPolicy<ExecSpace>& policy) {
      using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

      Kokkos::parallel_for(
          "sortByPt2", hintLightWeight(policy), KOKKOS_LAMBDA(const member_type& team_member) {
            sortByPt2(vdata, vws, team_member);
          });
      Kokkos::deep_copy(execSpace, hdata, vdata);
      execSpace.fence();

      auto& __restrict__ data = *hdata.data();
      uint32_t const& nvFinal = data.nvFinal;
      float* __restrict__ ptv2 = data.ptv2;
      uint16_t* __restrict__ sortInd = data.sortInd;

      // TODO: Kokkos::sort doesn't supported user-defined comparisons for now. A better
      // solution is to replace BinOp1D (in kokkos/algorithms/src/Kokkos_Sort.hpp) with
      // a custom comparison and create a sort function upon it.
      for (uint16_t i = 0; i < nvFinal; ++i)
        sortInd[i] = i;
      std::sort(sortInd, sortInd + nvFinal, [&](auto i, auto j) { return ptv2[i] < ptv2[j]; });
    }

  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
