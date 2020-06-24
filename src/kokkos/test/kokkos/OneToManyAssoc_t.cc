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
using Assoc = OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;

using SmallAssoc = OneToManyAssoc<uint16_t, 128, MaxAssocs>;

using Multiplicity = OneToManyAssoc<uint16_t, 8, MaxTk>;

using TeamPolicy = Kokkos::TeamPolicy<KokkosExecSpace>;
using MemberType = TeamPolicy::member_type;
using TeamView = Kokkos::View<Multiplicity::CountersOnly*,KokkosExecSpace::scratch_memory_space,Kokkos::MemoryUnmanaged>;
  
//using TK = Kokkos::View<uint16_t**,KokkosExecSpace>; std::array<uint16_t, 4>;

void countMultiLocal(Kokkos::View<uint16_t**,KokkosExecSpace> const tk,
                     Kokkos::View<Multiplicity*,KokkosExecSpace> assoc,
                     const int32_t& n) {
  auto nThreads = 256;
  auto nBlocks = (4 * n + nThreads - 1) / nThreads;
  
  TeamPolicy policy(nBlocks,nThreads);
  auto team_view_size = TeamView::shmem_size(1);
  auto level = 0;
  Kokkos::parallel_for("countMultiLocal",policy.set_scratch_size(level,Kokkos::PerTeam(team_view_size)),
    KOKKOS_LAMBDA(const MemberType teamMember){
      TeamView local(teamMember.team_scratch(level),1);
      if(teamMember.team_rank() == 0)
        local(0).zero();
      auto first = teamMember.team_size() * teamMember.league_rank() + teamMember.team_rank();

      for (int i = first; i < n; i += teamMember.league_size() * teamMember.team_size()) {
        teamMember.team_barrier();
        local(0).countDirect(2 + i % 4);
        teamMember.team_barrier();
        if(teamMember.team_rank() == 0)
          assoc(0).add(local);
      }
    });

}

template <typename T>
void countMulti(Kokkos::View<T*,KokkosExecSpace> assoc,const uint32_t& n) {
  Kokkos::parallel_for("countMulti",Kokkos::RangePolicy<KokkosExecSpace>(0,n),
    KOKKOS_LAMBDA(const int& i){
      assoc(0).countDirect(2 + i % 4);
    });
}

void verifyMulti(Kokkos::View<Multiplicity*,KokkosExecSpace> m1,
                 Kokkos::View<Multiplicity*,KokkosExecSpace> m2) {
  Kokkos::parallel_for("verifyMulti",Kokkos::RangePolicy<KokkosExecSpace>(0,Multiplicity::totbins()),
    KOKKOS_LAMBDA(const int& i){
      assert(m1(0).off[i] == m2(0).off[i]);
    });
}

void count(Kokkos::View<uint16_t**,KokkosExecSpace> const tk,
           Kokkos::View<Assoc*,KokkosExecSpace> assoc,
           const uint32_t& n) {

  Kokkos::parallel_for("count",Kokkos::RangePolicy<KokkosExecSpace>(0,4*n),
    KOKKOS_LAMBDA(const int& i){
      uint32_t k = i / 4;
      auto j = i - 4 * k;
      assert(j < 4);
      if (k >= n)
        return;
      if (tk(k,j) < MaxElem){
        printf("11 tk[%03d][%03d] = %06d\n",k,j,tk(k,j));
        assoc(0).countDirect(tk(k,j));
      }
    });
}

void fill(Kokkos::View<uint16_t**,KokkosExecSpace> const tk,
          Kokkos::View<Assoc*,KokkosExecSpace> assoc,
          const uint32_t& n) {
  Kokkos::parallel_for("fill",Kokkos::RangePolicy<KokkosExecSpace>(0,4*n),
    KOKKOS_LAMBDA(const int& i){
      uint32_t k = i / 4;
      auto j = i - 4 * k;
      assert(j < 4);
      if (k >= n)
        return;
      if (tk(k,j) < MaxElem)
        assoc(0).fillDirect(tk(k,j), k);
    });
}

void verify(Kokkos::View<Assoc*,Kokkos::HostSpace> assoc) { assert(assoc(0).size() < Assoc::capacity()); }

template <typename Assoc>
void fillBulk(Kokkos::View<AtomicPairCounter*,KokkosExecSpace> apc,
              Kokkos::View<uint16_t**,KokkosExecSpace> const tk, 
              Kokkos::View<Assoc*,KokkosExecSpace> assoc, 
              const uint32_t& n) {
  Kokkos::parallel_for("fillBulk",Kokkos::RangePolicy<KokkosExecSpace>(0,4*n),
    KOKKOS_LAMBDA(const int& i){
      auto m = tk(i,3) < MaxElem ? 4 : 3;
      assoc(0).bulkFill(apc, &tk(i,0), m);
    });
}

template<typename HistoType>
void verifyBulk(Kokkos::View<HistoType*,Kokkos::HostSpace> assoc, Kokkos::View<AtomicPairCounter*,Kokkos::HostSpace> apc) {
  if (apc(0).get().m >= HistoType::nbins())
    printf("Overflow %d %d\n", apc(0).get().m, HistoType::nbins());
  assert(assoc(0).size() < HistoType::capacity());
}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize,1024*1024*1024);

  std::cout << "OneToManyAssoc " << Assoc::nbins() << ' ' << Assoc::capacity() << ' ' << Assoc::wsSize() << std::endl;
  std::cout << "OneToManyAssoc (small) " << SmallAssoc::nbins() << ' ' << SmallAssoc::capacity() << ' '
            << SmallAssoc::wsSize() << std::endl;

  std::mt19937 eng;

  std::geometric_distribution<int> rdm(0.8);

  constexpr uint32_t N = 4000;

  Kokkos::View<uint16_t**,KokkosExecSpace> v_d("v_d",N,4);
  auto v_h = Kokkos::create_mirror_view(v_d);
  Kokkos::View<Assoc*,KokkosExecSpace> a_d("a_d",1);
  auto a_h = Kokkos::create_mirror_view(a_d);
  Kokkos::View<SmallAssoc*,KokkosExecSpace> sa_d("sa_d",1);
  auto sa_h = Kokkos::create_mirror_view(sa_d);
  Kokkos::View<AtomicPairCounter*,KokkosExecSpace> dc_d("dc_d",1);
  auto dc_h = Kokkos::create_mirror_view(dc_d);


  // fill with "index" to element
  long long ave = 0;
  int imax = 0;
  auto n = 0U;
  auto z = 0U;
  auto nz = 0U;
  for (auto i = 0U; i < 4U; ++i) {
    auto j = 0U;
    while (j < N && n < MaxElem) {
      if (z == 11) {
        ++n;
        z = 0;
        ++nz;
        continue;
      }  // a bit of not assoc
      auto x = rdm(eng);
      auto k = std::min(j + x + 1, N);
      if (i == 3 && z == 3) {  // some triplets time to time
        for (; j < k; ++j)
          v_h(j,i) = MaxElem + 1;
      } else {
        ave += x + 1;
        imax = std::max(imax, x);
        for (; j < k; ++j)
          v_h(j,i) = n;
        ++n;
      }
      ++z;
    }
    assert(n <= MaxElem);
    assert(j <= N);
  }
  std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;
  Kokkos::deep_copy(v_d,v_h);
  
  cms::kokkos::launchZero(a_d);
  Kokkos::deep_copy(a_h,a_d);
  for( uint32_t i = 0; i < a_h(0).totbins();++i)
    printf("0 a[%06d] = %06d\n",i,a_h(0).off[i]);

  // auto nThreads = 256;
  // auto nBlocks = (4 * N + nThreads - 1) / nThreads;

  count(v_d,a_d,N);
  Kokkos::deep_copy(a_h,a_d);
  for( uint32_t i = 0; i < a_h(0).totbins();++i)
    printf("1 a[%06d] = %06d\n",i,a_h(0).off[i]);
  
  cms::kokkos::launchFinalize(a_d);

  Kokkos::deep_copy(a_h,a_d);
  for( uint32_t i = 0; i < a_h(0).totbins();++i)
    printf("2 a[%06d] = %06d\n",i,a_h(0).off[i]);

  verify(a_h);

  fill(v_d, a_d, N);

  Kokkos::deep_copy(a_h,a_d);

  std::cout << a_h(0).size() << std::endl;
  imax = 0;
  ave = 0;
  z = 0;
  for (auto i = 0U; i < n; ++i) {
    auto x = a_h(0).size(i);
    if (x == 0) {
      z++;
      continue;
    }
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == a_h(0).size(n));
  std::cout << "found with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << z << std::endl;

  // now the inverse map (actually this is the direct....)
  
  fillBulk(dc_d, v_d, a_d, N);
  
  cms::kokkos::finalizeBulk(dc_d, a_d,Assoc::totbins());
  
  Kokkos::deep_copy(a_h,a_d);
  Kokkos::deep_copy(dc_h,dc_d);
  verifyBulk<Assoc>(a_h, dc_h);

  fillBulk(dc_d, v_d, sa_d, N);
  
  cms::kokkos::finalizeBulk(dc_d, sa_d,Assoc::totbins());
  
  Kokkos::deep_copy(sa_h,sa_d);
  Kokkos::deep_copy(dc_h,dc_d);
  verifyBulk<SmallAssoc>(sa_h, dc_h);
  
  std::cout << "final counter value " << dc_h(0).get().n << ' ' << dc_h(0).get().m << std::endl;

  std::cout << a_h(0).size() << std::endl;

  imax = 0;
  ave = 0;
  for (auto i = 0U; i < N; ++i) {
    auto x = a_h(0).size(i);
    if (!(x == 4 || x == 3))
      std::cout << i << ' ' << x << std::endl;
    assert(x == 4 || x == 3);
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == a_h(0).size(N));
  std::cout << "found with ave occupancy " << double(ave) / N << ' ' << imax << std::endl;

  // here verify use of block local counters
  Kokkos::View<Multiplicity*,KokkosExecSpace> m1_d("m1_d",1);
  Kokkos::View<Multiplicity*,KokkosExecSpace> m2_d("m2_d",1);

  cms::kokkos::launchZero(m1_d);
  cms::kokkos::launchZero(m2_d);

  // nBlocks = (4 * N + nThreads - 1) / nThreads;
  countMulti(m1_d, N);

  auto nThreads = 256;
  auto nBlocks = (4 * N + nThreads - 1) / nThreads;
  
  TeamPolicy policy(nBlocks,nThreads);
  auto team_view_size = TeamView::shmem_size(1);

  countMultiLocal(v_d, m2_d, N);
  

  verifyMulti(m1_d, m2_d);

  cms::kokkos::launchFinalize(m1_d);
  cms::kokkos::launchFinalize(m2_d);
  verifyMulti(m1_d, m2_d);


  return 0;
}
