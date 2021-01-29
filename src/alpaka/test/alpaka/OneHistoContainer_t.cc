#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include "AlpakaCore/HistoContainer.h"

template <int NBINS, int S, int DELTA>
struct mykernel {
  template <typename T_Acc, typename T>
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

    const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
      cms::alpakatools::element_global_index_range_uncut(acc, Vec1::all(Hist::totbins()));

    // set off zero
    uint32_t endElementIdx0 = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < Hist::totbins(); threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx0, Hist::totbins()); ++j) {
	hist.off[j] = 0;
      }
      endElementIdx0 += blockDimension;
    }
    alpaka::block::sync::syncBlockThreads(acc);

    // set bins zero
    uint32_t endElementIdx1 = endElementIdxNoStride[0u];    
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < Hist::capacity(); threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx1, Hist::totbins()); ++j) {
	hist.bins[j] = 0;
      }
      endElementIdx1 += blockDimension;
    }
    alpaka::block::sync::syncBlockThreads(acc);

    // count
    uint32_t endElementIdx2 = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < N; threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx2, N); ++j) {
        hist.count(acc, v[j]);
      }
      endElementIdx2 += blockDimension;
    }
    alpaka::block::sync::syncBlockThreads(acc);

    assert(0 == hist.size());
    alpaka::block::sync::syncBlockThreads(acc);

    // finalize
    hist.finalize(acc, ws);
    alpaka::block::sync::syncBlockThreads(acc);

    if (threadIdxLocal == 0) {
      printf("hist.size() = %u.\n", hist.size());
    }
    assert(N == hist.size());

    // verify
    uint32_t endElementIdx8 = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < Hist::nbins(); threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx8, Hist::nbins()); ++j) {
        assert(hist.off[j] <= hist.off[j + 1]);
      }
      endElementIdx8 += blockDimension;
    }
    alpaka::block::sync::syncBlockThreads(acc);

    if (threadIdxLocal < 32) {
      ws[threadIdxLocal] = 0;  // used by prefix scan...
    }
    alpaka::block::sync::syncBlockThreads(acc);

    // fill
    uint32_t endElementIdx3 = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < N; threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx3, N); ++j) {
        hist.fill(acc, v[j], j);
      }
      endElementIdx3 += blockDimension;
    }
    alpaka::block::sync::syncBlockThreads(acc);

    assert(0 == hist.off[0]);
    assert(N == hist.size());

    // bin
    uint32_t endElementIdx4 = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < hist.size() - 1; threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx4, hist.size() - 1); ++j) {
        auto p = hist.begin() + j;
        assert((*p) < N);
        auto k1 = Hist::bin(v[*p]);
        auto k2 = Hist::bin(v[*(p + 1)]);
        assert(k2 >= k1);
      }
      endElementIdx4 += blockDimension;
    }

    // forEachInWindow
    uint32_t endElementIdx5 = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < hist.size(); threadIdx += blockDimension) {
      for (uint32_t i = threadIdx; i < std::min(endElementIdx5, hist.size()); ++i) {
	auto p = hist.begin() + i;
	auto j = *p;
	auto b0 = Hist::bin(v[j]);
	int tot = 0;
	auto ftest = [&](unsigned int k) {
	  assert(k >= 0 && k < N);
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
      endElementIdx5 += blockDimension;
    }

  }
};


template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go(const DevHost& host, const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& device, ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) {
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
  auto v_dbuf = alpaka::mem::buf::alloc<T, Idx>(device, N);

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;


    alpaka::mem::view::copy(queue, v_dbuf, v_hbuf, N);

    const Vec1& threadsPerBlockOrElementsPerThread(Vec1::all(256));
    const Vec1& blocksPerGrid(Vec1::all(1));
    const WorkDiv1 &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDiv,
												mykernel<NBINS,S,DELTA>(),
												alpaka::mem::view::getPtrNative(v_dbuf),
												N
												));
    
  }
  alpaka::wait::wait(queue);
}


int main() {
  const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
  const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1 device(
						     alpaka::pltf::getDevByIdx<ALPAKA_ACCELERATOR_NAMESPACE::PltfAcc1>(0u));
  ALPAKA_ACCELERATOR_NAMESPACE::Queue queue(device);

  go<int16_t>(host, device, queue);
  go<uint8_t, 128, 8, 4>(host, device, queue);
  go<uint16_t, 313 / 2, 9, 4>(host, device, queue);

  return 0;
}
