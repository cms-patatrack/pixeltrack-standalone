#include "kokkosAlgo2.h"

namespace {
  constexpr int NUM_VALUES = 1000;

  KOKKOS_INLINE_FUNCTION void vectorProd(const Kokkos::View<float*, KokkosExecSpace> a,
                                         const Kokkos::View<float*, KokkosExecSpace> b,
                                         Kokkos::View<float**, KokkosExecSpace> c,
                                         size_t row,
                                         size_t col) {
    c(row, col) = a[row] * b[col];
  }
}  // namespace

namespace KOKKOS_NAMESPACE {
  Kokkos::View<float*, KokkosExecSpace> kokkosAlgo2() {
    Kokkos::View<float*, KokkosExecSpace> d_a{"d_a", NUM_VALUES};
    Kokkos::View<float*, KokkosExecSpace> d_b{"d_b", NUM_VALUES};

    auto h_a = Kokkos::create_mirror_view(d_a);
    auto h_b = Kokkos::create_mirror_view(d_b);

    for (int i = 0; i < NUM_VALUES; i++) {
      h_a[i] = i;
      h_b[i] = i * i;
    }

    Kokkos::deep_copy(d_a, h_a);
    Kokkos::deep_copy(d_b, h_b);

    Kokkos::View<float*, KokkosExecSpace> d_c{"d_c", NUM_VALUES};
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, NUM_VALUES),
        KOKKOS_LAMBDA(const size_t i) { d_c[i] = d_a[i] + d_b[i]; });

    Kokkos::View<float**, KokkosExecSpace> d_ma{"d_ma", NUM_VALUES, NUM_VALUES};
    Kokkos::View<float**, KokkosExecSpace> d_mb{"d_mb", NUM_VALUES, NUM_VALUES};
    Kokkos::View<float**, KokkosExecSpace> d_mc{"d_mc", NUM_VALUES, NUM_VALUES};

    auto policy = Kokkos::MDRangePolicy<KokkosExecSpace, Kokkos::Rank<2>>({{0, 0}}, {{NUM_VALUES, NUM_VALUES}});
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t row, const size_t col) { vectorProd(d_a, d_b, d_ma, row, col); });
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t row, const size_t col) { vectorProd(d_a, d_c, d_mb, row, col); });
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t row, const size_t col) {
          float tmp = 0;
          for (int i = 0; i < NUM_VALUES; ++i) {
            tmp += d_ma(row, i) * d_mb(i, col);
          }
          d_mc(row, col) = tmp;
        });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, NUM_VALUES), KOKKOS_LAMBDA(const size_t row) {
          float tmp = 0;
          for (int i = 0; i < NUM_VALUES; ++i) {
            tmp += d_ma(row, i) * d_b[i];
          }
          d_c[row] = tmp;
        });
    return d_a;
  }
}  // namespace KOKKOS_NAMESPACE
