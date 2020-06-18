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

  std::default_random_engine generator (1234);
  std::uniform_int_distribution<T> rgen(rmin, rmax);

  constexpr uint32_t N = 20000;
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
    
    using TeamHist = HistoContainer<T, NBINS, N, S, uint16_t>;

    Kokkos::View<TeamHist*,KokkosExecSpace> histo_d("histo_d",1);
    auto histo_h = Kokkos::create_mirror_view(histo_d);

    Kokkos::parallel_for("set_zero",Kokkos::RangePolicy<KokkosExecSpace>(0,TeamHist::totbins()),
      KOKKOS_LAMBDA(const int& i){
        histo_d(0).off[i] = 0;
      });

    Kokkos::parallel_for("set_zero_bin",Kokkos::RangePolicy<KokkosExecSpace>(0,TeamHist::capacity()),
      KOKKOS_LAMBDA(const int& i){
        histo_d(0).bins[i] = 0;
      });

    Kokkos::parallel_for("count",Kokkos::RangePolicy<KokkosExecSpace>(0,N),
      KOKKOS_LAMBDA(const int& i){
        histo_d(0).count(v_d(i));
      });

    Kokkos::deep_copy(histo_h,histo_d);
    assert(histo_h(0).off[-1] == 0);
    assert(0 == histo_h(0).size());

    TeamHist::finalize(histo_d);

    Kokkos::deep_copy(histo_h,histo_d);

    assert(histo_h(0).off[-1] == histo_h(0).off[-2]);
    assert(N == histo_h(0).size());

    Kokkos::parallel_for("assert_check",Kokkos::RangePolicy<KokkosExecSpace>(0,TeamHist::totbins()-1),
      KOKKOS_LAMBDA(const int& i){
        assert(histo_d(0).off[i] <= histo_d(0).off[i+1]);
      });

    Kokkos::parallel_for("fill",Kokkos::RangePolicy<KokkosExecSpace>(0,N),
      KOKKOS_LAMBDA(const int& i){
        histo_d(0).fill(v_d(i),i);
      });

    Kokkos::deep_copy(histo_h,histo_d);

    assert(0 == histo_h(0).off[0]);
    assert(N == histo_h(0).size());

    Kokkos::parallel_for("bin",Kokkos::RangePolicy<KokkosExecSpace>(0,N-1),
      KOKKOS_LAMBDA(const int& i){
        auto p = histo_d(0).begin() + i;
        assert( (*p) < N);
        auto k1 = TeamHist::bin(v_d(*p));
        auto k2 = TeamHist::bin(v_d(*(p+1)));
        assert(k2 >= k1);
      });

    Kokkos::parallel_for("forEachInWindow",Kokkos::RangePolicy<KokkosExecSpace>(0,N),
      KOKKOS_LAMBDA(const int& i){
        auto p = histo_d(0).begin() + i;
        auto j = *p;
        auto b0 = TeamHist::bin(v_d(j));
        int tot = 0;
        auto ftest = [&](int k){
          assert(k >= 0 && k < N);
          ++tot;
        };
        forEachInWindow(histo_d,v_d(j), v_d(j), ftest);
        int rtot = histo_d(0).size(b0);

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
        forEachInWindow(histo_d, vm, vp, ftest);
        int bp = TeamHist::bin(vp);
        int bm = TeamHist::bin(vm);
        rtot = histo_d(0).end(bp) - histo_d(0).begin(bm);
        assert(tot == rtot);


      });

  }
}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  // cudaDeviceSetLimit(cudaLimitPrintfFifoSize,1024*1024*1024);
  go<int16_t>();
  go<uint8_t, 128, 8, 4>();
  go<uint16_t, 313 / 2, 9, 4>();

  return 0;
}
