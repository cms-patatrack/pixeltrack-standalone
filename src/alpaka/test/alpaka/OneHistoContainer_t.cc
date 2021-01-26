#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include "AlpakaCore/HistoContainer.h"

struct mykernel {
  template <typename T_Acc, typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
  ALPAKA_FN_ACC void operator()(const T_Acc &acc, T const* __restrict__ v, uint32_t N) const {
    assert(v);
    assert(N == 12000);

    const uint32_t threadIdxLocal(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    if (threadIdxLocal == 0) {
      printf("start kernel for %d data\n", N);
    }

    using Hist = cms::alpakatools::HistoContainer<T, NBINS, 12000, S, uint16_t>;

    auto&& hist = alpaka::block::shared::st::allocVar<Hist, __COUNTER__>(acc);
    auto&& ws = alpaka::block::shared::st::allocVar<typename Hist::Counter[32], __COUNTER__>(acc);

    const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

    const auto& [firstElementIdxTotBins, endElementIdxTotBins] =
      cms::alpakatools::element_global_index_range(acc, Vec1::all(Hist::totbins()));
    for (uint32_t j = firstElementIdxTotBins[0u]; j < endElementIdxTotBins[0u]; j += blockDimension) {
      hist.off[j] = 0;
    }
    alpaka::block::sync::syncBlockThreads(acc);

    const auto& [firstElementIdxCapacity, endElementIdxCapacity] =
      cms::alpakatools::element_global_index_range(acc, Vec1::all(Hist::capacity()));
    for (uint32_t j = firstElementIdxCapacity[0u]; j < endElementIdxCapacity[0u]; j += blockDimension) {
      hist.off[j] = 0;
    }
    alpaka::block::sync::syncBlockThreads(acc);

    const auto& [firstElementIdxN, endElementIdxN] =
      cms::alpakatools::element_global_index_range(acc, Vec1::all(N));
    for (uint32_t j = firstElementIdxN[0u]; j < endElementIdxN[0u]; j += blockDimension) {
      hist.count(acc, v[j]);
    }
    alpaka::block::sync::syncBlockThreads(acc);

    assert(0 == hist.size());
    alpaka::block::sync::syncBlockThreads(acc);

    hist.finalize(acc, ws);
    alpaka::block::sync::syncBlockThreads(acc);

    if (threadIdxLocal == 0) {
      printf("hist.size() = %u.\n", hist.size());
    }
    //assert(N == hist.size());
    const auto& [firstElementIdxNBins, endElementIdxNBins] =
      cms::alpakatools::element_global_index_range(acc, Vec1::all(Hist::nbins()));
    for (uint32_t j = firstElementIdxNBins[0u]; j < endElementIdxNBins[0u]; j += blockDimension) {
      assert(hist.off[j] <= hist.off[j + 1]);
    }
    alpaka::block::sync::syncBlockThreads(acc);

    if (threadIdxLocal < 32) {
      ws[threadIdxLocal] = 0;  // used by prefix scan...
    }
    alpaka::block::sync::syncBlockThreads(acc);

    for (uint32_t j = firstElementIdxN[0u]; j < endElementIdxN[0u]; j += blockDimension) {
      hist.fill(acc, v[j], j);
    }
    alpaka::block::sync::syncBlockThreads(acc);
    assert(0 == hist.off[0]);
    //assert(N == hist.size());

    const auto& [firstElementIdxSizeMinus1, endElementIdxSizeMinus1] =
      cms::alpakatools::element_global_index_range(acc, Vec1::all(hist.size() - 1));
    for (uint32_t j = firstElementIdxSizeMinus1[0u]; j < endElementIdxSizeMinus1[0u]; j += blockDimension) {
      auto p = hist.begin() + j;
      assert((*p) < N);
      auto k1 = Hist::bin(v[*p]);
      auto k2 = Hist::bin(v[*(p + 1)]);
      assert(k2 >= k1);
    }

    const auto& [firstElementIdxSize, endElementIdxSize] =
      cms::alpakatools::element_global_index_range(acc, Vec1::all(hist.size()));
    for (uint32_t i = firstElementIdxSize[0u]; i < endElementIdxSize[0u]; i += blockDimension) {
      auto p = hist.begin() + i;
      auto j = *p;
      auto b0 = Hist::bin(v[j]);
      int tot = 0;
      auto ftest = [&](unsigned int k) {
	assert(k < N);
	++tot;
      };
      cms::alpakatools::forEachInWindow(hist, v[j], v[j], ftest);
      int rtot = hist.size(b0);
      assert(tot == rtot);
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
      int bp = Hist::bin(vp);
      int bm = Hist::bin(vm);
      rtot = hist.end(bp) - hist.begin(bm);
      assert(tot == rtot);
    }

  }
};

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go(const DevHost& host, const DevAcc1& device, Queue& queue) {
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

  auto v_hbuf = alpaka::mem::buf::alloc<T, Idx>(host, N);
  auto v = alpaka::mem::view::getPtrNative(v_hbuf);

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;


    auto v_dbuf = alpaka::mem::buf::alloc<T, Idx>(device, N);
    alpaka::mem::view::copy(queue, v_dbuf, v_hbuf, N);
  
    const Vec1& threadsPerBlockOrElementsPerThread(Vec1::all(256));
    const Vec1& blocksPerGrid(Vec1::all(1));
    const WorkDiv1 &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<Acc1>(workDiv,
								  mykernel(),
								  alpaka::mem::view::getPtrNative(v_dbuf),
								  N
								  ));
    alpaka::wait::wait(queue);
    //launch(mykernel<T, NBINS, S, DELTA>, {1, 256}, v_d.get(), N);
  }
}

int main() {
  const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
  const DevAcc1 device(alpaka::pltf::getDevByIdx<PltfAcc1>(0u));
  Queue queue(device);

  go<int16_t>(host, device, queue);
  //go<uint8_t, 128, 8, 4>(host, device, queue);
  //go<uint16_t, 313 / 2, 9, 4>(host, device, queue);

  return 0;
}
