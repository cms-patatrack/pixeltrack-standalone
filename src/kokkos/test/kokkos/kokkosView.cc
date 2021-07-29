#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/memoryTraits.h"

#include <iostream>

constexpr int ELEMENTS = 10;

template <typename ExecSpace>
KOKKOS_INLINE_FUNCTION void process(const Kokkos::View<int*, ExecSpace, Restrict>& data, int i) {
  data(i) = 2 * i;
}

void test() {
  Kokkos::View<int*, KokkosExecSpace, Restrict> data_d("data_d", ELEMENTS);
  auto data_h = Kokkos::create_mirror_view(data_d);
  for (int i = 0; i < ELEMENTS; ++i) {
    data_h[i] = i;
  }
  Kokkos::deep_copy(KokkosExecSpace(), data_d, data_h);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, ELEMENTS),
      KOKKOS_LAMBDA(int i) { process(data_d, i); });
  Kokkos::deep_copy(KokkosExecSpace(), data_h, data_d);
  Kokkos::fence();
  for (int i = 0; i < ELEMENTS; ++i) {
    std::cout << i << " " << data_h(i) << std::endl;
  }
}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  test();
  return 0;
}
