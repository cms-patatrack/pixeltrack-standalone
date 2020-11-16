#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

#include <cassert>
#include <iostream>
#include "KokkosCore/AtomicPairCounter.h"

typedef Kokkos::TeamPolicy<KokkosExecSpace> team_policy;
typedef Kokkos::TeamPolicy<KokkosExecSpace>::member_type member_type;

void test() {
  Kokkos::View<cms::kokkos::AtomicPairCounter *, KokkosExecSpace> dc_d("dc_d", 1);
  Kokkos::View<cms::kokkos::AtomicPairCounter *, KokkosExecSpace>::HostMirror dc_h("dc_h", 1);

  std::cout << "size " << sizeof(cms::kokkos::AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;

  Kokkos::View<uint32_t *, KokkosExecSpace> n_d("n_d", N);
  Kokkos::View<uint32_t *, KokkosExecSpace> m_d("m_d", M);

  const uint32_t n = 10000;
  team_policy local_policy(n, Kokkos::AUTO());

  std::cout << "league size = " << local_policy.league_size() << " team size = " << local_policy.team_size()
            << std::endl;
  Kokkos::parallel_for(
      "update", local_policy, KOKKOS_LAMBDA(const member_type &teamMember) {
        uint32_t i = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank();
        if (i >= n)
          return;

        auto m = i % 11;
        m = m % 6 + 1;  // max 6, no 0
        auto c = dc_d(0).add(m);
        assert(c.m < n);
        n_d[c.m] = c.n;
        for (uint32_t j = c.n; j < c.n + m; ++j)
          m_d[j] = i;
      });

  Kokkos::parallel_for(
      "finalize", team_policy(1, 1), KOKKOS_LAMBDA(const member_type &teamMember) {
        assert(dc_d(0).get().m == n);
        n_d[n] = dc_d(0).get().n;
      });

  Kokkos::parallel_for(
      "verify", local_policy, KOKKOS_LAMBDA(const member_type &teamMember) {
        uint32_t i = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank();
        if (i >= n)
          return;
        assert(0 == n_d[0]);
        assert(dc_d(0).get().m == n);
        assert(n_d[n] == dc_d(0).get().n);
        auto ib = n_d[i];
        auto ie = n_d[i + 1];
        auto k = m_d[ib++];
        assert(k < n);
        for (; ib < ie; ++ib)
          assert(m_d[ib] == k);
      });

  Kokkos::deep_copy(KokkosExecSpace(), dc_h, dc_d);

  std::cout << dc_h(0).get().n << ' ' << dc_h(0).get().m << std::endl;

}  // test()

int main(void) {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  test();
  return 0;
}
