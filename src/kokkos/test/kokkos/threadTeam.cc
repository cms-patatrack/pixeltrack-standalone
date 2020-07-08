#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

#include <iostream>

namespace {
  constexpr int TEAMS = 10;
  constexpr int THREADS_PER_TEAM = 128;
  constexpr int ELEMENTS_PER_TEAM = 1000;
  constexpr int ELEMENTS = TEAMS * ELEMENTS_PER_TEAM;

  KOKKOS_INLINE_FUNCTION void kernel(Kokkos::View<int*, KokkosExecSpace> data,
                                     Kokkos::TeamPolicy<KokkosExecSpace>::member_type const& teamMember) {
    //printf("%d %d %d\n", static_cast<int>(teamMember.league_rank()), static_cast<int>(teamMember.team_size()), static_cast<int>(teamMember.team_rank()));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, ELEMENTS_PER_TEAM), [=](int i) {
      data[teamMember.league_rank() * ELEMENTS_PER_TEAM + i] *= 10;
      //printf("%d %d %d\n", static_cast<int>(teamMember.league_rank()), static_cast<int>(teamMember.team_size()), i);
    });
    /*
    for (int i=0; i<ELEMENTS_PER_TEAM; i += teamMember.team_size()) {
      data[teamMember.league_rank()*ELEMENTS_PER_TEAM + i] *= 10;
    }
    */
  }
}  // namespace

void test() {
  Kokkos::View<int*, KokkosExecSpace> data_d("data_d", ELEMENTS);
  auto data_h = Kokkos::create_mirror_view(data_d);
  for (int i = 0; i < ELEMENTS; ++i) {
    data_h[i] = i;
  }
  Kokkos::deep_copy(KokkosExecSpace(), data_d, data_h);

  using TeamPolicy = Kokkos::TeamPolicy<KokkosExecSpace>;
  using MemberType = TeamPolicy::member_type;
  //TeamPolicy policy(KokkosExecSpace(), TEAMS, THREADS_PER_TEAM);
  TeamPolicy policy(KokkosExecSpace(), TEAMS, Kokkos::AUTO());
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(MemberType const& teamMember) { kernel(data_d, teamMember); });

  Kokkos::deep_copy(KokkosExecSpace(), data_h, data_d);

  KokkosExecSpace().fence();
  for (int iTeam = 0; iTeam != TEAMS; ++iTeam) {
    for (int i = 0; i < 10; ++i) {
      std::cout << "Team " << iTeam << " element " << i << " " << data_h[iTeam * ELEMENTS_PER_TEAM + i] << std::endl;
    }
  }
}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  test();
  return 0;
}
