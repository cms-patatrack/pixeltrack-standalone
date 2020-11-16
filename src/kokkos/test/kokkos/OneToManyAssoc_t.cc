#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include <array>
#include <memory>

#include "KokkosCore/HistoContainer.h"

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;
using Assoc = cms::kokkos::OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;

using SmallAssoc = cms::kokkos::OneToManyAssoc<uint16_t, 128, MaxAssocs>;

using Multiplicity = cms::kokkos::OneToManyAssoc<uint16_t, 8, MaxTk>;

using TeamPolicy = Kokkos::TeamPolicy<KokkosExecSpace>;
using MemberType = TeamPolicy::member_type;
using TeamView =
    Kokkos::View<Multiplicity::CountersOnly, KokkosExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;

//using TK = Kokkos::View<uint16_t**,KokkosExecSpace>; std::array<uint16_t, 4>;

template <typename ExecSpace>
void countMultiLocal(Kokkos::View<uint16_t**, ExecSpace> const tk,
                     Kokkos::View<Multiplicity, ExecSpace> assoc,
                     const int32_t& n,
                     ExecSpace const& execSpace) {
  auto nThreads = 256;
  auto nBlocks = (4 * n + nThreads - 1) / nThreads;

#ifndef KOKKOS_BACKEND_SERIAL
  TeamPolicy policy(execSpace, nBlocks, nThreads);
#else
  TeamPolicy policy(execSpace, nBlocks * nThreads, 1);
#endif
  auto team_view_size = TeamView::shmem_size();
  auto level = 0;
  Kokkos::parallel_for(
      "countMultiLocal",
      policy.set_scratch_size(level, Kokkos::PerTeam(team_view_size)),
      KOKKOS_LAMBDA(const MemberType teamMember) {
        TeamView local(teamMember.team_scratch(level));
        if (teamMember.team_rank() == 0)
          local().zero();
        auto first = teamMember.team_size() * teamMember.league_rank() + teamMember.team_rank();

        for (int i = first; i < n; i += teamMember.league_size() * teamMember.team_size()) {
          teamMember.team_barrier();
          local().countDirect(2 + i % 4);
          teamMember.team_barrier();
          if (teamMember.team_rank() == 0)
            assoc().add(local);
        }
      });
}

template <typename T, typename ExecSpace>
void countMulti(Kokkos::View<T, ExecSpace> assoc, const uint32_t& n, ExecSpace const& execSpace) {
  Kokkos::parallel_for(
      "countMulti", Kokkos::RangePolicy<ExecSpace>(execSpace, 0, n), KOKKOS_LAMBDA(const int& i) {
        assoc().countDirect(2 + i % 4);
      });
}

template <typename ExecSpace>
void verifyMulti(Kokkos::View<Multiplicity, ExecSpace> m1,
                 Kokkos::View<Multiplicity, ExecSpace> m2,
                 ExecSpace const& execSpace) {
  Kokkos::parallel_for(
      "verifyMulti",
      Kokkos::RangePolicy<ExecSpace>(execSpace, 0, Multiplicity::totbins()),
      KOKKOS_LAMBDA(const int& i) { assert(m1().off[i] == m2().off[i]); });
}

template <typename ExecSpace>
void count(Kokkos::View<uint16_t**, ExecSpace> const tk,
           Kokkos::View<Assoc, ExecSpace> assoc,
           const uint32_t& n,
           ExecSpace const& execSpace) {
  Kokkos::parallel_for(
      "count", Kokkos::RangePolicy<ExecSpace>(execSpace, 0, 4 * n), KOKKOS_LAMBDA(const int& i) {
        uint32_t k = i / 4;
        auto j = i - 4 * k;
        assert(j < 4);
        if (k >= n)
          return;
        if (tk(k, j) < MaxElem) {
          //printf("11 tk[%03d][%03d] = %06d\n",k,j,tk(k,j));
          assoc().countDirect(tk(k, j));
        }
      });
}

template <typename ExecSpace>
void fill(Kokkos::View<uint16_t**, ExecSpace> const tk,
          Kokkos::View<Assoc, ExecSpace> assoc,
          const uint32_t& n,
          ExecSpace const& execSpace) {
  Kokkos::parallel_for(
      "fill", Kokkos::RangePolicy<ExecSpace>(execSpace, 0, 4 * n), KOKKOS_LAMBDA(const int& i) {
        uint32_t k = i / 4;
        auto j = i - 4 * k;
        assert(j < 4);
        if (k >= n)
          return;
        if (tk(k, j) < MaxElem)
          assoc().fillDirect(tk(k, j), k);
      });
}

template <typename ArrayLayout, typename ExecSpace>
void verify(Kokkos::View<Assoc, ArrayLayout, ExecSpace> assoc) {
  assert(assoc().size() < Assoc::capacity());
}

template <typename Assoc, typename ExecSpace>
void fillBulk(Kokkos::View<cms::kokkos::AtomicPairCounter, ExecSpace> apc,
              Kokkos::View<uint16_t**, ExecSpace> const tk,
              Kokkos::View<Assoc, ExecSpace> assoc,
              const uint32_t& n,
              ExecSpace const& execSpace) {
  Kokkos::parallel_for(
      "fillBulk", Kokkos::RangePolicy<ExecSpace>(execSpace, 0, n), KOKKOS_LAMBDA(const int& i) {
        auto m = tk(i, 3) < MaxElem ? 4 : 3;
        // printf("01 i = %06d m = %06d tk(i,3) = %06d tk(i,0) = %06d\n",i,m,tk(i,3),tk(i,0));
        auto x = assoc().bulkFill(apc, &tk(i, 0), m);
        // printf("01  x = %06d\n",x);
      });
}

template <typename HistoType, typename ArrayLayout, typename ExecSpace>
void verifyBulk(Kokkos::View<HistoType, ArrayLayout, ExecSpace> assoc,
                Kokkos::View<cms::kokkos::AtomicPairCounter, ArrayLayout, ExecSpace> apc) {
  if (apc().get().m >= HistoType::nbins())
    printf("Overflow %d %d\n", apc().get().m, HistoType::nbins());
  if (assoc().size() >= HistoType::capacity())
    printf("assert will fail: size = %06d capacity = %06d\n", assoc().size(), HistoType::capacity());
  assert(assoc().size() < HistoType::capacity());
}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
#ifdef KOKKOS_BACKEND_CUDA
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 1024);
#endif

  std::cout << "OneToManyAssoc " << Assoc::nbins() << ' ' << Assoc::capacity() << ' ' << Assoc::wsSize() << std::endl;
  std::cout << "OneToManyAssoc (small) " << SmallAssoc::nbins() << ' ' << SmallAssoc::capacity() << ' '
            << SmallAssoc::wsSize() << std::endl;

  std::mt19937 eng;
  std::default_random_engine generator(1234);
  std::geometric_distribution<int> rdm(0.8);
  constexpr uint32_t rdm_N = 16000;
  Kokkos::View<int*, KokkosExecSpace> rdm_d("rdm_d", rdm_N);
  auto rdm_h = Kokkos::create_mirror_view(rdm_d);
  for (uint32_t i = 0; i < rdm_N; ++i) {
    rdm_h(i) = rdm(eng);
    //printf("00 rdm(eng) = %04d\n",rdm_h(i));
  }
  Kokkos::deep_copy(KokkosExecSpace(), rdm_d, rdm_h);

  constexpr uint32_t N = 4000;

  Kokkos::View<uint16_t**, KokkosExecSpace> v_d("v_d", N, 4);
  auto v_h = Kokkos::create_mirror_view(v_d);
  Kokkos::View<Assoc, KokkosExecSpace> a_d("a_d");
  auto a_h = Kokkos::create_mirror_view(a_d);
  Kokkos::View<SmallAssoc, KokkosExecSpace> sa_d("sa_d");
  auto sa_h = Kokkos::create_mirror_view(sa_d);
  Kokkos::View<cms::kokkos::AtomicPairCounter, KokkosExecSpace> dc_d("dc_d");
  auto dc_h = Kokkos::create_mirror_view(dc_d);

  Kokkos::View<long long, KokkosExecSpace> ave_d("ave_d");
  auto ave_h = Kokkos::create_mirror_view(ave_d);
  Kokkos::View<int, KokkosExecSpace> imax_d("imax_d");
  auto imax_h = Kokkos::create_mirror_view(imax_d);
  Kokkos::View<uint32_t, KokkosExecSpace> n_d("n_d");
  auto n_h = Kokkos::create_mirror_view(n_d);
  Kokkos::View<uint32_t, KokkosExecSpace> z_d("z_d");
  auto z_h = Kokkos::create_mirror_view(z_d);
  Kokkos::View<uint32_t, KokkosExecSpace> nz_d("nz_d");
  auto nz_h = Kokkos::create_mirror_view(nz_d);
  // fill with "index" to element
  Kokkos::parallel_for(
      "init", Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 1), KOKKOS_LAMBDA(const int& i) {
        ave_d() = 0;
        imax_d() = 0;
        n_d() = 0;
        z_d() = 0;
        nz_d() = 0;

        uint32_t rdm_counter = 0;

        for (auto i = 0U; i < 4U; ++i) {
          auto j = 0U;
          while (j < N && n_d() < MaxElem) {
            if (z_d() == 11) {
              n_d() += 1;
              z_d() = 0;
              nz_d() += 1;
              continue;
            }  // a bit of not assoc
            auto x = rdm_d(rdm_counter);
            //printf("00 i = %04d j = %04d rdm(eng) = %04d\n",i,j,x);
            rdm_counter++;
            if (rdm_counter >= rdm_N)
              printf("ERROR ran out of random numbers\n");
            auto k = N;
            if (j + x + 1 < N)
              k = j + x + 1;
            if (i == 3 && z_d() == 3) {  // some triplets time to time
              for (; j < k; ++j)
                v_d(j, i) = MaxElem + 1;
            } else {
              ave_d() += x + 1;
              if (x > imax_d())
                imax_d() = x;
              for (; j < k; ++j)
                v_d(j, i) = n_d();
              n_d() += 1;
            }
            z_d() += 1;
          }
          assert(n_d() <= MaxElem);
          assert(j <= N);
        }
      });

  Kokkos::deep_copy(KokkosExecSpace(), n_h, n_d);
  Kokkos::deep_copy(KokkosExecSpace(), nz_h, nz_d);
  Kokkos::deep_copy(KokkosExecSpace(), ave_h, ave_d);
  Kokkos::deep_copy(KokkosExecSpace(), imax_h, imax_d);

  std::cout << "filled with " << n_h() << " elements " << double(ave_h()) / n_h() << ' ' << imax_h() << ' ' << nz_h()
            << std::endl;
  // Kokkos::deep_copy(v_d,v_h);

  cms::kokkos::launchZero(a_d, KokkosExecSpace());

  // auto nThreads = 256;
  // auto nBlocks = (4 * N + nThreads - 1) / nThreads;

  count(v_d, a_d, N, KokkosExecSpace());

  cms::kokkos::launchFinalize(a_d, KokkosExecSpace());

  Kokkos::deep_copy(KokkosExecSpace(), a_h, a_d);

  verify(a_h);

  fill(v_d, a_d, N, KokkosExecSpace());

  Kokkos::deep_copy(KokkosExecSpace(), a_h, a_d);
  std::cout << a_h().size() << std::endl;
  Kokkos::deep_copy(KokkosExecSpace(), n_h, n_d);
  imax_h() = 0;
  ave_h() = 0;
  z_h() = 0;
  for (auto i = 0U; i < n_h(); ++i) {
    auto x = a_h().size(i);
    if (x == 0) {
      z_h()++;
      continue;
    }
    ave_h() += x;
    imax_h() = std::max(imax_h(), int(x));
  }
  assert(0 == a_h().size(n_h()));
  std::cout << "found with " << n_h() << " elements " << double(ave_h()) / n_h() << ' ' << imax_h() << ' ' << z_h()
            << std::endl;

  // now the inverse map (actually this is the direct....)
  dc_h().zero();
  Kokkos::deep_copy(KokkosExecSpace(), dc_d, dc_h);
  fillBulk(dc_d, v_d, a_d, N, KokkosExecSpace());

  cms::kokkos::finalizeBulk(dc_d, a_d, KokkosExecSpace());

  Kokkos::deep_copy(KokkosExecSpace(), a_h, a_d);
  Kokkos::deep_copy(KokkosExecSpace(), dc_h, dc_d);
  verifyBulk<Assoc>(a_h, dc_h);

  // zero out counter
  dc_h().zero();
  Kokkos::deep_copy(KokkosExecSpace(), dc_d, dc_h);

  fillBulk(dc_d, v_d, sa_d, N, KokkosExecSpace());

  cms::kokkos::finalizeBulk(dc_d, sa_d, KokkosExecSpace());

  Kokkos::deep_copy(KokkosExecSpace(), sa_h, sa_d);
  Kokkos::deep_copy(KokkosExecSpace(), dc_h, dc_d);
  verifyBulk<SmallAssoc>(sa_h, dc_h);

  std::cout << "final counter value " << dc_h().get().n << ' ' << dc_h().get().m << std::endl;

  std::cout << a_h().size() << std::endl;

  imax_h() = 0;
  ave_h() = 0;
  for (auto i = 0U; i < N; ++i) {
    auto x = a_h().size(i);
    if (!(x == 4 || x == 3))
      std::cout << i << ' ' << x << std::endl;
    assert(x == 4 || x == 3);
    ave_h() += x;
    imax_h() = std::max(imax_h(), int(x));
  }
  assert(0 == a_h().size(N));
  std::cout << "found with ave occupancy " << double(ave_h()) / N << ' ' << imax_h() << std::endl;

  // here verify use of block local counters
  Kokkos::View<Multiplicity, KokkosExecSpace> m1_d("m1_d");
  Kokkos::View<Multiplicity, KokkosExecSpace> m2_d("m2_d");

  cms::kokkos::launchZero(m1_d, KokkosExecSpace());
  cms::kokkos::launchZero(m2_d, KokkosExecSpace());

  // nBlocks = (4 * N + nThreads - 1) / nThreads;
  countMulti(m1_d, N, KokkosExecSpace());

  countMultiLocal(v_d, m2_d, N, KokkosExecSpace());

  verifyMulti(m1_d, m2_d, KokkosExecSpace());

  cms::kokkos::launchFinalize(m1_d, KokkosExecSpace());
  cms::kokkos::launchFinalize(m2_d, KokkosExecSpace());
  verifyMulti(m1_d, m2_d, KokkosExecSpace());

  return 0;
}
