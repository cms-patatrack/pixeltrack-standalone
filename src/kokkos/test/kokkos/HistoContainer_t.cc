#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <random>

#include "KokkosCore/HistoContainer.h"

template <typename T>
void go() {
  std::mt19937 eng;
  std::uniform_int_distribution<T> rgen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  constexpr int N = 12000;
  Kokkos::View<T*, KokkosExecSpace> v_d("v_d", N);
  auto v_h = Kokkos::create_mirror_view(v_d);

  constexpr uint32_t nParts = 10;
  constexpr uint32_t partSize = N / nParts;
  //  uint32_t offsets[nParts + 1];

  using Hist = cms::kokkos::HistoContainer<T, 128, N, 8 * sizeof(T), uint32_t, nParts>;
  std::cout << "HistoContainer " << (int)(offsetof(Hist, off)) << ' ' << Hist::nbins() << ' ' << Hist::totbins() << ' '
            << Hist::capacity() << ' ' << Hist::wsSize() << ' '
            << (std::numeric_limits<T>::max() - std::numeric_limits<T>::min()) / Hist::nbins() << std::endl;

  Kokkos::View<Hist, KokkosExecSpace> h_d("h_d");
  auto h_h = Kokkos::create_mirror_view(h_d);

  Kokkos::View<uint32_t*, KokkosExecSpace> off_d("off_d", nParts + 1);
  auto off_h = Kokkos::create_mirror_view(off_d);

  for (int it = 0; it < 5; ++it) {
    off_h(0) = 0;
    for (uint32_t j = 1; j < nParts + 1; ++j) {
      off_h(j) = off_h(j - 1) + partSize - 3 * j;
      assert(off_h(j) <= N);
    }

    if (it == 1) {  // special cases...
      off_h(0) = 0;
      off_h(1) = 0;
      off_h(2) = 19;
      off_h(3) = 32 + off_h(2);
      off_h(4) = 123 + off_h(3);
      off_h(5) = 256 + off_h(4);
      off_h(6) = 311 + off_h(5);
      off_h(7) = 2111 + off_h(6);
      off_h(8) = 256 * 11 + off_h(7);
      off_h(9) = 44 + off_h(8);
      off_h(10) = 3297 + off_h(9);
    }
    Kokkos::deep_copy(KokkosExecSpace(), off_d, off_h);

    for (long long j = 0; j < N; j++)
      v_h[j] = rgen(eng);

    if (it == 2) {  // big bin
      for (long long j = 1000; j < 2000; j++)
        v_h[j] = sizeof(T) == 1 ? 22 : 3456;
    }
    Kokkos::deep_copy(KokkosExecSpace(), v_d, v_h);

    cms::kokkos::fillManyFromVector(h_d,
                                    nParts,
                                    Kokkos::View<T const*, KokkosExecSpace>(v_d),
                                    Kokkos::View<uint32_t const*, KokkosExecSpace>(off_d),
                                    off_h(10),
                                    256,
                                    KokkosExecSpace());

    Kokkos::deep_copy(KokkosExecSpace(), h_h, h_d);
    auto h = h_h();

    assert(0 == h.off[0]);
    assert(off_h(10) == h.size());

    auto verify = [&](uint32_t i, uint32_t k, uint32_t t1, uint32_t t2) {
      assert(static_cast<int>(t1) < N);
      assert(static_cast<int>(t2) < N);
      if (T(v_h(t1) - v_h(t2)) <= 0)
        std::cout << "for " << i << ':' << v_h(k) << " failed " << v_h(t1) << ' ' << v_h(t2) << std::endl;
    };

    auto incr = [](auto& k) { return k = (k + 1) % Hist::nbins(); };

    // make sure it spans 3 bins...
    auto window = T(1300);

    for (uint32_t j = 0; j < nParts; ++j) {
      auto off = Hist::histOff(j);
      for (uint32_t i = 0; i < Hist::nbins(); ++i) {
        auto ii = i + off;
        if (0 == h.size(ii))
          continue;
        auto k = *h.begin(ii);
        if (j % 2)
          k = *(h.begin(ii) + (h.end(ii) - h.begin(ii)) / 2);
        auto bk = h.bin(v_h(k));
        assert(bk == i);
        assert(k < off_h(j + 1));
        auto kl = h.bin(v_h(k) - window);
        auto kh = h.bin(v_h(k) + window);
        assert(kl != i);
        assert(kh != i);
        // std::cout << kl << ' ' << kh << std::endl;

        auto me = v_h(k);
        auto tot = 0;
        auto nm = 0;
        bool l = true;
        auto khh = kh;
        incr(khh);
        for (auto kk = kl; kk != khh; incr(kk)) {
          if (kk != kl && kk != kh)
            nm += h.size(kk + off);
          for (auto p = h.begin(kk + off); p < h.end(kk + off); ++p) {
            if (std::min(std::abs(T(v_h(*p) - me)), std::abs(T(me - v_h(*p)))) > window) {
            } else {
              ++tot;
            }
          }
          if (kk == i) {
            l = false;
            continue;
          }
          if (l)
            for (auto p = h.begin(kk + off); p < h.end(kk + off); ++p)
              verify(i, k, k, (*p));
          else
            for (auto p = h.begin(kk + off); p < h.end(kk + off); ++p)
              verify(i, k, (*p), k);
        }
        if (!(tot >= nm)) {
          std::cout << "too bad " << j << ' ' << i << ' ' << int(me) << '/' << (int)T(me - window) << '/'
                    << (int)T(me + window) << ": " << kl << '/' << kh << ' ' << khh << ' ' << tot << '/' << nm
                    << std::endl;
        }
        if (l)
          std::cout << "what? " << j << ' ' << i << ' ' << int(me) << '/' << (int)T(me - window) << '/'
                    << (int)T(me + window) << ": " << kl << '/' << kh << ' ' << khh << ' ' << tot << '/' << nm
                    << std::endl;
        assert(!l);
      }
    }
  }
}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  go<int16_t>();
  go<int8_t>();

  return 0;
}
