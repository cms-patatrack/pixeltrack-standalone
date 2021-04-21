#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include <array>
#include <memory>

#include "AlpakaCore/HistoContainer.h"

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;

using Assoc = cms::alpakatools::OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;
using SmallAssoc = cms::alpakatools::OneToManyAssoc<uint16_t, 128, MaxAssocs>;
using Multiplicity = cms::alpakatools::OneToManyAssoc<uint16_t, 8, MaxTk>;
using TK = std::array<uint16_t, 4>;

struct countMultiLocal {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                TK const* __restrict__ tk,
                                Multiplicity* __restrict__ assoc,
                                uint32_t n) const {
    cms::alpakatools::for_each_element_in_grid_strided(acc, n, [&](uint32_t i) {
      auto& local = alpaka::declareSharedVar<Multiplicity::CountersOnly, __COUNTER__>(acc);
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const bool oncePerSharedMemoryAccess = (threadIdxLocal == 0);
      if (oncePerSharedMemoryAccess) {
        local.zero();
      }
      alpaka::syncBlockThreads(acc);
      local.countDirect(acc, 2 + i % 4);
      alpaka::syncBlockThreads(acc);
      if (oncePerSharedMemoryAccess) {
        assoc->add(acc, local);
      }
    });
  }
};

struct countMulti {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                TK const* __restrict__ tk,
                                Multiplicity* __restrict__ assoc,
                                uint32_t n) const {
    cms::alpakatools::for_each_element_in_grid_strided(acc, n, [&](uint32_t i) { assoc->countDirect(acc, 2 + i % 4); });
  }
};

struct verifyMulti {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Multiplicity* __restrict__ m1, Multiplicity* __restrict__ m2) const {
    cms::alpakatools::for_each_element_in_grid_strided(
        acc, Multiplicity::totbins(), [&](uint32_t i) { assert(m1->off[i] == m2->off[i]); });
  }
};

struct count {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                TK const* __restrict__ tk,
                                Assoc* __restrict__ assoc,
                                uint32_t n) const {
    cms::alpakatools::for_each_element_in_grid_strided(acc, 4 * n, [&](uint32_t i) {
      auto k = i / 4;
      auto j = i - 4 * k;
      assert(j < 4);
      if (k >= n) {
        return;
      }
      if (tk[k][j] < MaxElem) {
        assoc->countDirect(acc, tk[k][j]);
      }
    });
  }
};

struct fill {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                TK const* __restrict__ tk,
                                Assoc* __restrict__ assoc,
                                uint32_t n) const {
    cms::alpakatools::for_each_element_in_grid_strided(acc, 4 * n, [&](uint32_t i) {
      auto k = i / 4;
      auto j = i - 4 * k;
      assert(j < 4);
      if (k >= n) {
        return;
      }
      if (tk[k][j] < MaxElem) {
        assoc->fillDirect(acc, tk[k][j], k);
      }
    });
  }
};

struct verify {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Assoc* __restrict__ assoc) const {
    assert(assoc->size() < Assoc::capacity());
  }
};

struct fillBulk {
  template <typename T_Acc, typename Assoc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                cms::alpakatools::AtomicPairCounter* apc,
                                TK const* __restrict__ tk,
                                Assoc* __restrict__ assoc,
                                uint32_t n) const {
    cms::alpakatools::for_each_element_in_grid_strided(acc, n, [&](uint32_t k) {
      auto m = tk[k][3] < MaxElem ? 4 : 3;
      assoc->bulkFill(acc, *apc, &tk[k][0], m);
    });
  }
};

struct verifyBulk {
  template <typename T_Acc, typename Assoc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                Assoc const* __restrict__ assoc,
                                cms::alpakatools::AtomicPairCounter const* apc) const {
    if (apc->get().m >= Assoc::nbins()) {
      printf("Overflow %d %d\n", apc->get().m, Assoc::nbins());
    }
    assert(assoc->size() < Assoc::capacity());
  }
};

int main() {
  const DevHost host(alpaka::getDevByIdx<PltfHost>(0u));
  const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1 device(alpaka::getDevByIdx<ALPAKA_ACCELERATOR_NAMESPACE::PltfAcc1>(0u));
  ALPAKA_ACCELERATOR_NAMESPACE::Queue queue(device);

  std::cout << "OneToManyAssoc " << sizeof(Assoc) << ' ' << Assoc::nbins() << ' ' << Assoc::capacity() << std::endl;
  std::cout << "OneToManyAssoc (small) " << sizeof(SmallAssoc) << ' ' << SmallAssoc::nbins() << ' '
            << SmallAssoc::capacity() << std::endl;

  std::mt19937 eng;
  std::geometric_distribution<int> rdm(0.8);

  constexpr uint32_t N = 4000;

  auto tr_hbuf = alpaka::allocBuf<std::array<uint16_t, 4>, Idx>(host, N);
  auto tr = alpaka::getPtrNative(tr_hbuf);
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
          tr[j][i] = MaxElem + 1;
      } else {
        ave += x + 1;
        imax = std::max(imax, x);
        for (; j < k; ++j)
          tr[j][i] = n;
        ++n;
      }
      ++z;
    }
    assert(n <= MaxElem);
    assert(j <= N);
  }
  std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;

  auto v_dbuf = alpaka::allocBuf<std::array<uint16_t, 4>, Idx>(device, N);
  alpaka::memcpy(queue, v_dbuf, tr_hbuf, N);

  auto a_dbuf = alpaka::allocBuf<Assoc, Idx>(device, 1u);
  alpaka::memset(queue, a_dbuf, 0, 1u);

  const unsigned int nThreads = 256;
  const Vec1 threadsPerBlockOrElementsPerThread(nThreads);
  const unsigned int nBlocks4N = (4 * N + nThreads - 1) / nThreads;
  const Vec1 blocksPerGrid4N(nBlocks4N);
  const WorkDiv1& workDiv4N = cms::alpakatools::make_workdiv(blocksPerGrid4N, threadsPerBlockOrElementsPerThread);

  launchZero(alpaka::getPtrNative(a_dbuf), queue);

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      workDiv4N, count(), alpaka::getPtrNative(v_dbuf), alpaka::getPtrNative(a_dbuf), N));

  cms::alpakatools::launchFinalize(alpaka::getPtrNative(a_dbuf), queue);

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      WorkDiv1{Vec1::all(1u), Vec1::all(1u), Vec1::all(1u)}, verify(), alpaka::getPtrNative(a_dbuf)));

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      workDiv4N, fill(), alpaka::getPtrNative(v_dbuf), alpaka::getPtrNative(a_dbuf), N));

  auto la_hbuf = alpaka::allocBuf<Assoc, Idx>(host, 1u);
  alpaka::memcpy(queue, la_hbuf, a_dbuf, 1u);
  alpaka::wait(queue);

  auto la = alpaka::getPtrNative(la_hbuf);

  std::cout << la->size() << std::endl;
  imax = 0;
  ave = 0;
  z = 0;
  for (auto i = 0U; i < n; ++i) {
    auto x = la->size(i);
    if (x == 0) {
      z++;
      continue;
    }
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la->size(n));
  std::cout << "found with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << z << std::endl;

  // now the inverse map (actually this is the direct....)
  auto dc_dbuf = alpaka::allocBuf<cms::alpakatools::AtomicPairCounter, Idx>(device, 1u);
  alpaka::memset(queue, dc_dbuf, 0, 1u);

  const unsigned int nBlocks = (N + nThreads - 1) / nThreads;
  const Vec1 blocksPerGrid(nBlocks);
  const WorkDiv1& workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDiv,
                                                                               fillBulk(),
                                                                               alpaka::getPtrNative(dc_dbuf),
                                                                               alpaka::getPtrNative(v_dbuf),
                                                                               alpaka::getPtrNative(a_dbuf),
                                                                               N));

  alpaka::enqueue(
      queue,
      alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
          workDiv, cms::alpakatools::finalizeBulk(), alpaka::getPtrNative(dc_dbuf), alpaka::getPtrNative(a_dbuf)));

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      WorkDiv1{Vec1::all(1u), Vec1::all(1u), Vec1::all(1u)},
                      verifyBulk(),
                      alpaka::getPtrNative(a_dbuf),
                      alpaka::getPtrNative(dc_dbuf)));

  alpaka::memcpy(queue, la_hbuf, a_dbuf, 1u);

  auto dc_hbuf = alpaka::allocBuf<cms::alpakatools::AtomicPairCounter, Idx>(host, 1u);
  alpaka::memcpy(queue, dc_hbuf, dc_dbuf, 1u);
  alpaka::wait(queue);
  auto dc = alpaka::getPtrNative(dc_hbuf);

  alpaka::memset(queue, dc_dbuf, 0, 1u);
  auto sa_dbuf = alpaka::allocBuf<SmallAssoc, Idx>(device, 1u);
  alpaka::memset(queue, sa_dbuf, 0, 1u);

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDiv,
                                                                               fillBulk(),
                                                                               alpaka::getPtrNative(dc_dbuf),
                                                                               alpaka::getPtrNative(v_dbuf),
                                                                               alpaka::getPtrNative(sa_dbuf),
                                                                               N));

  alpaka::enqueue(
      queue,
      alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
          workDiv, cms::alpakatools::finalizeBulk(), alpaka::getPtrNative(dc_dbuf), alpaka::getPtrNative(sa_dbuf)));

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      WorkDiv1{Vec1::all(1u), Vec1::all(1u), Vec1::all(1u)},
                      verifyBulk(),
                      alpaka::getPtrNative(sa_dbuf),
                      alpaka::getPtrNative(dc_dbuf)));

  std::cout << "final counter value " << dc->get().n << ' ' << dc->get().m << std::endl;

  std::cout << la->size() << std::endl;
  imax = 0;
  ave = 0;
  for (auto i = 0U; i < N; ++i) {
    auto x = la->size(i);
    if (!(x == 4 || x == 3)) {
      std::cout << i << ' ' << x << std::endl;
    }
    assert(x == 4 || x == 3);
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la->size(N));
  std::cout << "found with ave occupancy " << double(ave) / N << ' ' << imax << std::endl;

  // here verify use of block local counters
  auto m1_dbuf = alpaka::allocBuf<Multiplicity, Idx>(device, 1u);
  alpaka::memset(queue, m1_dbuf, 0, 1u);
  auto m2_dbuf = alpaka::allocBuf<Multiplicity, Idx>(device, 1u);
  alpaka::memset(queue, m2_dbuf, 0, 1u);

  launchZero(alpaka::getPtrNative(m1_dbuf), queue);
  launchZero(alpaka::getPtrNative(m2_dbuf), queue);

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      workDiv4N, countMulti(), alpaka::getPtrNative(v_dbuf), alpaka::getPtrNative(m1_dbuf), N));

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      workDiv4N, countMultiLocal(), alpaka::getPtrNative(v_dbuf), alpaka::getPtrNative(m2_dbuf), N));

  const Vec1 blocksPerGridTotBins(1u);
  const Vec1 threadsPerBlockOrElementsPerThreadTotBins(Multiplicity::totbins());
  const WorkDiv1& workDivTotBins =
      cms::alpakatools::make_workdiv(blocksPerGridTotBins, threadsPerBlockOrElementsPerThreadTotBins);

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      workDivTotBins, verifyMulti(), alpaka::getPtrNative(m1_dbuf), alpaka::getPtrNative(m2_dbuf)));

  cms::alpakatools::launchFinalize(alpaka::getPtrNative(m1_dbuf), queue);
  cms::alpakatools::launchFinalize(alpaka::getPtrNative(m2_dbuf), queue);

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                      workDivTotBins, verifyMulti(), alpaka::getPtrNative(m1_dbuf), alpaka::getPtrNative(m2_dbuf)));

  alpaka::wait(queue);

  return 0;
}
