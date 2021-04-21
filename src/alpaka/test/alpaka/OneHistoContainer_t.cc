#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include "AlpakaCore/HistoContainer.h"

template <int NBINS, int S, int DELTA>
struct mykernel {
  template <typename T_Acc, typename T>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, T const* __restrict__ v, uint32_t N) const {
    assert(v);
    assert(N == 12000);

    const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    if (threadIdxLocal == 0) {
      printf("start kernel for %d data\n", N);
    }

    using Hist = cms::alpakatools::HistoContainer<T, NBINS, 12000, S, uint16_t>;

    auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
    auto& ws = alpaka::declareSharedVar<typename Hist::Counter[32], __COUNTER__>(acc);

    // set off zero
    cms::alpakatools::for_each_element_in_block_strided(acc, Hist::totbins(), [&](uint32_t j) { hist.off[j] = 0; });
    alpaka::syncBlockThreads(acc);

    // set bins zero
    cms::alpakatools::for_each_element_in_block_strided(acc, Hist::totbins(), [&](uint32_t j) { hist.bins[j] = 0; });
    alpaka::syncBlockThreads(acc);

    // count
    cms::alpakatools::for_each_element_in_block_strided(acc, N, [&](uint32_t j) { hist.count(acc, v[j]); });
    alpaka::syncBlockThreads(acc);

    assert(0 == hist.size());
    alpaka::syncBlockThreads(acc);

    // finalize
    hist.finalize(acc, ws);
    alpaka::syncBlockThreads(acc);

    assert(N == hist.size());

    // verify
    cms::alpakatools::for_each_element_in_block_strided(
        acc, Hist::nbins(), [&](uint32_t j) { assert(hist.off[j] <= hist.off[j + 1]); });
    alpaka::syncBlockThreads(acc);

    cms::alpakatools::for_each_element_in_block(acc, 32, [&](uint32_t i) {
      ws[i] = 0;  // used by prefix scan...
    });
    alpaka::syncBlockThreads(acc);

    // fill
    cms::alpakatools::for_each_element_in_block_strided(acc, N, [&](uint32_t j) { hist.fill(acc, v[j], j); });
    alpaka::syncBlockThreads(acc);

    assert(0 == hist.off[0]);
    assert(N == hist.size());

    // bin
#ifndef NDEBUG
    cms::alpakatools::for_each_element_in_block_strided(acc, hist.size() - 1, [&](uint32_t j) {
      auto p = hist.begin() + j;
      assert((*p) < N);
      auto k1 = Hist::bin(v[*p]);
      auto k2 = Hist::bin(v[*(p + 1)]);
      assert(k2 >= k1);
    });
#endif

    // forEachInWindow
    cms::alpakatools::for_each_element_in_block_strided(acc, hist.size(), [&](uint32_t i) {
      auto p = hist.begin() + i;
      auto j = *p;
#ifndef NDEBUG
      auto b0 = Hist::bin(v[j]);
#endif
      int tot = 0;
      auto ftest = [&](unsigned int k) {
        assert(k < N);
        ++tot;
      };
      cms::alpakatools::forEachInWindow(hist, v[j], v[j], ftest);
#ifndef NDEBUG
      int rtot = hist.size(b0);
      assert(tot == rtot);
#endif
      tot = 0;
      auto vm = int(v[j]) - DELTA;
      auto vp = int(v[j]) + DELTA;
      constexpr int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
      vm = std::max(vm, 0);
      vm = std::min(vm, vmax);
      vp = std::min(vp, vmax);
      vp = std::max(vp, 0);
      assert(vp >= vm);
      cms::alpakatools::forEachInWindow(hist, vm, vp, ftest);
#ifndef NDEBUG
      int bp = Hist::bin(vp);
      int bm = Hist::bin(vm);
      rtot = hist.end(bp) - hist.begin(bm);
      assert(tot == rtot);
#endif
    });
  }
};

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go(const DevHost& host,
        const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& device,
        ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) {
  std::mt19937 eng;

  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::uniform_int_distribution<T> rgen(rmin, rmax);
  constexpr unsigned int N = 12000;

  using Hist = cms::alpakatools::HistoContainer<T, NBINS, N, S>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::capacity() << ' '
            << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;

  auto v_hbuf = alpaka::allocBuf<T, Idx>(host, N);
  auto v = alpaka::getPtrNative(v_hbuf);
  auto v_dbuf = alpaka::allocBuf<T, Idx>(device, N);

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;

    alpaka::memcpy(queue, v_dbuf, v_hbuf, N);

    const Vec1& threadsPerBlockOrElementsPerThread(Vec1::all(256));
    const Vec1& blocksPerGrid(Vec1::all(1));
    const WorkDiv1& workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                        workDiv, mykernel<NBINS, S, DELTA>(), alpaka::getPtrNative(v_dbuf), N));
  }
  alpaka::wait(queue);
}

int main() {
  const DevHost host(alpaka::getDevByIdx<PltfHost>(0u));
  const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1 device(alpaka::getDevByIdx<ALPAKA_ACCELERATOR_NAMESPACE::PltfAcc1>(0u));
  ALPAKA_ACCELERATOR_NAMESPACE::Queue queue(device);

  go<int16_t>(host, device, queue);
  go<uint8_t, 128, 8, 4>(host, device, queue);
  go<uint16_t, 313 / 2, 9, 4>(host, device, queue);

  return 0;
}
