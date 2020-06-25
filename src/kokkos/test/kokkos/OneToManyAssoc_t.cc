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
        //printf("11 tk[%03d][%03d] = %06d\n",k,j,tk(k,j));
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
  Kokkos::parallel_for("fillBulk",Kokkos::RangePolicy<KokkosExecSpace>(0,n),
    KOKKOS_LAMBDA(const int& i){
      auto m = tk(i,3) < MaxElem ? 4 : 3;
      // printf("01 i = %06d m = %06d tk(i,3) = %06d tk(i,0) = %06d\n",i,m,tk(i,3),tk(i,0));
      auto x = assoc(0).bulkFill(apc, &tk(i,0), m,i);
      // printf("01  x = %06d\n",x);
    });
}

template<typename HistoType>
void verifyBulk(Kokkos::View<HistoType*,Kokkos::HostSpace> assoc, Kokkos::View<AtomicPairCounter*,Kokkos::HostSpace> apc) {
  if (apc(0).get().m >= HistoType::nbins())
    printf("Overflow %d %d\n", apc(0).get().m, HistoType::nbins());
  if(assoc(0).size() >= HistoType::capacity())
    printf("assert will fail: size = %06d capacity = %06d\n",assoc(0).size(),HistoType::capacity());
  assert(assoc(0).size() < HistoType::capacity());
}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize,1024*1024*1024);

  std::cout << "OneToManyAssoc " << Assoc::nbins() << ' ' << Assoc::capacity() << ' ' << Assoc::wsSize() << std::endl;
  std::cout << "OneToManyAssoc (small) " << SmallAssoc::nbins() << ' ' << SmallAssoc::capacity() << ' '
            << SmallAssoc::wsSize() << std::endl;

  std::mt19937 eng;
  std::default_random_engine generator (1234);
  std::geometric_distribution<int> rdm(0.8);
  constexpr uint32_t rdm_N = 16000;
  Kokkos::View<int*,KokkosExecSpace> rdm_d("rdm_d",rdm_N);
  auto rdm_h = Kokkos::create_mirror_view(rdm_d);
  for(uint32_t i=0;i<rdm_N;++i){
    rdm_h(i) = rdm(eng);
    //printf("00 rdm(eng) = %04d\n",rdm_h(i));
  }
  Kokkos::deep_copy(rdm_d,rdm_h);

  constexpr uint32_t N = 4000;

  Kokkos::View<uint16_t**,KokkosExecSpace> v_d("v_d",N,4);
  auto v_h = Kokkos::create_mirror_view(v_d);
  Kokkos::View<Assoc*,KokkosExecSpace> a_d("a_d",1);
  auto a_h = Kokkos::create_mirror_view(a_d);
  Kokkos::View<SmallAssoc*,KokkosExecSpace> sa_d("sa_d",1);
  auto sa_h = Kokkos::create_mirror_view(sa_d);
  Kokkos::View<AtomicPairCounter*,KokkosExecSpace> dc_d("dc_d",1);
  auto dc_h = Kokkos::create_mirror_view(dc_d);


  Kokkos::View<long long*,KokkosExecSpace> ave_d("ave_d",1);
  auto ave_h = Kokkos::create_mirror_view(ave_d);
  Kokkos::View<int*,KokkosExecSpace> imax_d("imax_d",1);
  auto imax_h = Kokkos::create_mirror_view(imax_d);
  Kokkos::View<uint32_t*,KokkosExecSpace> n_d("n_d",1);
  auto n_h = Kokkos::create_mirror_view(n_d);
  Kokkos::View<uint32_t*,KokkosExecSpace> z_d("z_d",1);
  auto z_h = Kokkos::create_mirror_view(z_d);
  Kokkos::View<uint32_t*,KokkosExecSpace> nz_d("nz_d",1);
  auto nz_h = Kokkos::create_mirror_view(nz_d);
  // fill with "index" to element
  Kokkos::parallel_for("init",Kokkos::RangePolicy<KokkosExecSpace>(0,1),
    KOKKOS_LAMBDA(const int& i){
      ave_d(0) = 0;
      imax_d(0) = 0;
      n_d(0) = 0;
      z_d(0) = 0;
      nz_d(0) = 0;

      uint32_t rdm_counter = 0;

      for (auto i = 0U; i < 4U; ++i) {
        auto j = 0U;
        while (j < N && n_d(0) < MaxElem) {
          if (z_d(0) == 11) {
            n_d(0) += 1;
            z_d(0) = 0;
            nz_d(0) += 1;
            continue;
          }  // a bit of not assoc
          auto x = rdm_d(rdm_counter);
          //printf("00 i = %04d j = %04d rdm(eng) = %04d\n",i,j,x);
          rdm_counter++;
          if(rdm_counter>=rdm_N)
            printf("ERROR ran out of random numbers\n");
          auto k = N;
          if(j + x + 1 < N)
            k = j + x + 1;
          if (i == 3 && z_d(0) == 3) {  // some triplets time to time
            for (; j < k; ++j)
              v_d(j,i) = MaxElem + 1;
          } else {
            ave_d(0) += x + 1;
            if(x > imax_d(0))
              imax_d(0) = x;
            for (; j < k; ++j)
              v_d(j,i) = n_d(0);
            n_d(0) += 1;
          }
          z_d(0) += 1;
        }
        assert(n_d(0) <= MaxElem);
        assert(j <= N);
      }
    });

  Kokkos::deep_copy(n_h,n_d);
  Kokkos::deep_copy(nz_h,nz_d);
  Kokkos::deep_copy(ave_h,ave_d);
  Kokkos::deep_copy(imax_h,imax_d);

  std::cout << "filled with " << n_h(0) << " elements " << double(ave_h(0)) / n_h(0) << ' ' << imax_h(0) << ' ' << nz_h(0) << std::endl;
  // Kokkos::deep_copy(v_d,v_h);

  
  cms::kokkos::launchZero(a_d);

  // auto nThreads = 256;
  // auto nBlocks = (4 * N + nThreads - 1) / nThreads;

  count(v_d,a_d,N);
  
  cms::kokkos::launchFinalize(a_d);

  Kokkos::deep_copy(a_h,a_d);

  verify(a_h);

  fill(v_d, a_d, N);

  Kokkos::deep_copy(a_h,a_d);
  std::cout << a_h(0).size() << std::endl;
  Kokkos::deep_copy(n_h,n_d);
  imax_h(0) = 0;
  ave_h(0) = 0;
  z_h(0) = 0;
  for (auto i = 0U; i < n_h(0); ++i) {
    auto x = a_h(0).size(i);
    if (x == 0) {
      z_h(0)++;
      continue;
    }
    ave_h(0) += x;
    imax_h(0) = std::max(imax_h(0), int(x));
  }
  assert(0 == a_h(0).size(n_h(0)));
  std::cout << "found with " << n_h(0) << " elements " << double(ave_h(0)) / n_h(0) << ' ' << imax_h(0) << ' ' << z_h(0) << std::endl;

  // now the inverse map (actually this is the direct....)
  
  fillBulk(dc_d, v_d, a_d, N);
  
  cms::kokkos::finalizeBulk(dc_d, a_d);
  
  Kokkos::deep_copy(a_h,a_d);
  Kokkos::deep_copy(dc_h,dc_d);
  verifyBulk<Assoc>(a_h, dc_h);

  // zero out counter
  dc_h(0).zero();
  Kokkos::deep_copy(dc_d,dc_h);

  fillBulk(dc_d, v_d, sa_d, N);

  cms::kokkos::finalizeBulk(dc_d, sa_d);
  
  Kokkos::deep_copy(sa_h,sa_d);
  Kokkos::deep_copy(dc_h,dc_d);
  verifyBulk<SmallAssoc>(sa_h, dc_h);
  
  std::cout << "final counter value " << dc_h(0).get().n << ' ' << dc_h(0).get().m << std::endl;

  std::cout << a_h(0).size() << std::endl;

  imax_h(0) = 0;
  ave_h(0) = 0;
  for (auto i = 0U; i < N; ++i) {
    auto x = a_h(0).size(i);
    if (!(x == 4 || x == 3))
      std::cout << i << ' ' << x << std::endl;
    assert(x == 4 || x == 3);
    ave_h(0) += x;
    imax_h(0) = std::max(imax_h(0), int(x));
  }
  assert(0 == a_h(0).size(N));
  std::cout << "found with ave occupancy " << double(ave_h(0)) / N << ' ' << imax_h(0) << std::endl;

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
