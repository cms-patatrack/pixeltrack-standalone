#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include "KokkosCore/HistoContainer.h"
using team_policy = Kokkos::TeamPolicy<KokkosExecSpace>;
using member_type = Kokkos::TeamPolicy<KokkosExecSpace>::member_type;

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go() {
  std::mt19937 eng;

  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::uniform_int_distribution<T> rgen(rmin, rmax);

  constexpr int N = 12000;
  Kokkos::View<T*,KokkosExecSpace> v_d("v_d",N);
  typename Kokkos::View<T*,KokkosExecSpace>::HostMirror v_h("v_h",N);

  using Hist = HistoContainer<T, NBINS, N, S>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::capacity() << ' '
            << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v_h(j) = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v_h(j) = 4;

    Kokkos::deep_copy(v_d,v_h);
    team_policy policy(1,256);
    int level = 0;

    using TeamHist = HistoContainer<T, NBINS, N, S, uint16_t>;
    using TeamHistView = Kokkos::View<TeamHist*,KokkosExecSpace::scratch_memory_space,Kokkos::MemoryUnmanaged>;
    size_t team_hist_bytes = TeamHistView::shmem_size(1);

    using WSView = typename Kokkos::View<typename TeamHist::Counter*,KokkosExecSpace::scratch_memory_space,Kokkos::MemoryUnmanaged>;
    int ws_size = 32;
    size_t team_ws_bytes = WSView::shmem_size(ws_size);
    printf("starting loop: creating shared mem size: %10lu\n",team_hist_bytes+team_ws_bytes);
    Kokkos::parallel_for("mykernel",policy.set_scratch_size(level,Kokkos::PerTeam(team_hist_bytes+team_ws_bytes)),
                         KOKKOS_LAMBDA(const member_type& teamMember){

      uint32_t threadId = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank();
      printf("threadId = %10u",threadId);
      if (threadId == 0)
        printf("start kernel for %d data\n", N);

      TeamHistView shared_histo(teamMember.team_scratch(level),1);
      auto hist = shared_histo(0);
      WSView ws(teamMember.team_scratch(level),ws_size);

//      for (auto j = threadId; j < TeamHist::totbins(); j += teamMember.team_size()) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,TeamHist::totbins()),
         [&](const int& j){
        hist.off[j] = 0;
      });
      teamMember.team_barrier();
      Kokkos::single(Kokkos::PerTeam(teamMember), [&]{
        for(uint32_t i = 0;i<TeamHist::totbins();++i){
          printf("hist.off[%02u] = %04u\n",i,hist.off[i]);
        }
      });
#ifdef TODO
      for (auto j = threadId; j < N; j += teamMember.team_size())
        hist.count(v_d(j));
      teamMember.team_barrier();

      assert(0 == hist.size());
      teamMember.team_barrier();

      hist.finalize(&ws(0),teamMember);
      teamMember.team_barrier();

      assert(N == hist.size());
      for (auto j = threadId; j < Hist::nbins(); j += teamMember.team_size())
        assert(hist.off[j] <= hist.off[j + 1]);
      teamMember.team_barrier();

      if (threadId < 32)
        ws[threadId] = 0;  // used by prefix scan...
      teamMember.team_barrier();

      for (auto j = threadIdx.x; j < N; j += blockDim.x)
        hist.fill(v_d(j), j);
      teamMember.team_barrier();
      assert(0 == hist.off[0]);
      assert(N == hist.size());

      for (auto j = threadId; j < hist.size() - 1; j += teamMember.team_size()) {
        auto p = hist.begin() + j;
        assert((*p) < N);
        auto k1 = TeamHist::bin(v_d(*p));
        auto k2 = TeamHist::bin(v_d(*(p + 1)));
        assert(k2 >= k1);
      }

      for (auto i = threadId; i < hist.size(); i += teamMember.team_size()) {
        auto p = hist.begin() + i;
        auto j = *p;
        auto b0 = TeamHist::bin(v_d(j));
        int tot = 0;
        auto ftest = [&](int k) {
          assert(k >= 0 && k < N);
          ++tot;
        };
        forEachInWindow(hist, v_d(j), v_d(j), ftest);
        int rtot = hist.size(b0);
        assert(tot == rtot);
        tot = 0;
        auto vm = int(v_d(j)) - DELTA;
        auto vp = int(v_d(j)) + DELTA;
        constexpr int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
        vm = std::max(vm, 0);
        vm = std::min(vm, vmax);
        vp = std::min(vp, vmax);
        vp = std::max(vp, 0);
        assert(vp >= vm);
        forEachInWindow(hist, vm, vp, ftest);
        int bp = TeamHist::bin(vp);
        int bm = TeamHist::bin(vm);
        rtot = hist.end(bp) - hist.begin(bm);
        assert(tot == rtot);
      }
#endif // TODO

    }); // KOKKOS_LAMBDA
  }
}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard;
  go<int16_t>();
//  go<uint8_t, 128, 8, 4>();
//  go<uint16_t, 313 / 2, 9, 4>();

  return 0;
}
