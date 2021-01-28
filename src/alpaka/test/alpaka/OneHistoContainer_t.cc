#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include "AlpakaCore/HistoContainer.h"

struct setZero {
  template <typename T_Acc, typename Histo>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const T_Acc &acc, Histo *__restrict__ hist) const {
    const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
    const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(Histo::totbins()));
    uint32_t endElementIdx = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < Histo::totbins(); threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx, Histo::totbins()); ++j) {
        hist->off[j] = 0;
      }
      endElementIdx += blockDimension;
    }
  }
};

struct setZeroBins {
  template <typename T_Acc, typename Histo>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const T_Acc &acc, Histo *__restrict__ hist) const {
    const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
    const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(Histo::capacity()));
    uint32_t endElementIdx = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < Histo::capacity(); threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx, Histo::capacity()); ++j) {
        hist->bins[j] = 0;
      }
      endElementIdx += blockDimension;
    }
  }
};

struct count {
  template <typename T_Acc, typename Histo, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const T_Acc &acc, Histo *__restrict__ hist, T *v, uint32_t N) const {
    const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
    const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(N));
    uint32_t endElementIdx = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < N; threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx, N); ++j) {
        hist->count(acc, v[j]);
      }
      endElementIdx += blockDimension;
    }
  }
};

struct finalize {
  template <typename T_Acc, typename Histo>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const T_Acc &acc, Histo *__restrict__ hist) const {
    auto &&ws = alpaka::block::shared::st::allocVar<typename Histo::Counter[32], __COUNTER__>(acc);
    hist->finalize(acc, ws);
  }
};

struct verify {
  template <typename T_Acc, typename Histo>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const T_Acc &acc, Histo *__restrict__ hist) const {
    const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
    const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(Histo::nbins()));
    uint32_t endElementIdx = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < Histo::nbins(); threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx, Histo::nbins()); ++j) {
        assert(hist->off[j] <= hist->off[j + 1]);
      }
      endElementIdx += blockDimension;
    }
  }
};

struct fill {
  template <typename T_Acc, typename Histo, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const T_Acc &acc, Histo *__restrict__ hist, T *v, uint32_t N) const {
    const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
    const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(N));
    uint32_t endElementIdx = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < N; threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx, N); ++j) {
        hist->fill(acc, v[j], j);
      }
      endElementIdx += blockDimension;
    }
  }
};

struct bin {
  template <typename T_Acc, typename Histo, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(const T_Acc &acc, Histo *__restrict__ hist, T *v, uint32_t N) const {
    const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
    const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(hist->size() - 1));
    uint32_t endElementIdx = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < hist->size() - 1; threadIdx += blockDimension) {
      for (uint32_t j = threadIdx; j < std::min(endElementIdx, hist->size() - 1); ++j) {
        auto p = hist->begin() + j;
        assert((*p) < N);
        auto k1 = Histo::bin(v[*p]);
        auto k2 = Histo::bin(v[*(p + 1)]);
        assert(k2 >= k1);
      }
      endElementIdx += blockDimension;
    }
  }
};

struct forEachInWindow {
  template <typename T_Acc, typename Histo, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator()(
      const T_Acc &acc, Histo *__restrict__ hist, const T *v, uint32_t N, const int NBINS, const int DELTA) const {
    const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
    const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(hist->size()));
    uint32_t endElementIdx = endElementIdxNoStride[0u];
    for (uint32_t threadIdx = firstElementIdxNoStride[0u]; threadIdx < hist->size(); threadIdx += blockDimension) {
      for (uint32_t i = threadIdx; i < std::min(endElementIdx, hist->size()); ++i) {
        auto p = hist->begin() + i;
        auto j = *p;
        auto b0 = Histo::bin(v[j]);
        int tot = 0;
        auto ftest = [&](unsigned int k) {
          assert(k < N);
          ++tot;
        };
        cms::alpakatools::forEachInWindow(*hist, v[j], v[j], ftest);
        int rtot = hist->size(b0);
        assert(tot == rtot);
        tot = 0;
        auto vm = int(v[j]) - DELTA;
        auto vp = int(v[j]) + DELTA;
        const int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
        vm = std::max(vm, 0);
        vm = std::min(vm, vmax);
        vp = std::min(vp, vmax);
        vp = std::max(vp, 0);
        assert(vp >= vm);
        cms::alpakatools::forEachInWindow(*hist, vm, vp, ftest);
        int bp = Histo::bin(vp);
        int bm = Histo::bin(vm);
        rtot = hist->end(bp) - hist->begin(bm);
        assert(tot == rtot);
      }
      endElementIdx += blockDimension;
    }
  }
};

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go(const DevHost &host,
        const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1 &device,
        ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
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
    for (long long j = 0; j < N; j++) {
      v[j] = rgen(eng);
    }
    if (it == 2) {
      for (long long j = N / 2; j < N / 2 + N / 4; j++) {
        v[j] = 4;
      }
    }

    auto v_dbuf = alpaka::mem::buf::alloc<T, Idx>(device, N);
    alpaka::mem::view::copy(queue, v_dbuf, v_hbuf, N);

    printf("start kernel for %d data\n", N);

    using HistTeam = cms::alpakatools::HistoContainer<T, NBINS, N, S, uint16_t>;
    auto hist_hbuf = alpaka::mem::buf::alloc<HistTeam, Idx>(host, 1u);
    auto hist = alpaka::mem::view::getPtrNative(hist_hbuf);
    auto hist_dbuf = alpaka::mem::buf::alloc<HistTeam, Idx>(device, 1u);
    alpaka::mem::view::set(queue, hist_dbuf, 0, 1u);

    const Vec1 threadsPerBlockOrElementsPerThread(256u);
    const Vec1 blocksPerGrid(1u);
    const WorkDiv1 &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);

    alpaka::queue::enqueue(queue,
                           alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                               workDiv, setZero(), alpaka::mem::view::getPtrNative(hist_dbuf)));

    alpaka::queue::enqueue(queue,
                           alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                               workDiv, setZeroBins(), alpaka::mem::view::getPtrNative(hist_dbuf)));

    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
            workDiv, count(), alpaka::mem::view::getPtrNative(hist_dbuf), alpaka::mem::view::getPtrNative(v_dbuf), N));

    alpaka::mem::view::copy(queue, hist_hbuf, hist_dbuf, 1u);
    alpaka::wait::wait(queue);
    assert(0 == hist->size());

    alpaka::queue::enqueue(queue,
                           alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                               workDiv, finalize(), alpaka::mem::view::getPtrNative(hist_dbuf)));

    alpaka::mem::view::copy(queue, hist_hbuf, hist_dbuf, 1u);
    alpaka::wait::wait(queue);
    assert(N == hist->size());

    alpaka::queue::enqueue(queue,
                           alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                               workDiv, verify(), alpaka::mem::view::getPtrNative(hist_dbuf)));

    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
            workDiv, fill(), alpaka::mem::view::getPtrNative(hist_dbuf), alpaka::mem::view::getPtrNative(v_dbuf), N));

    alpaka::mem::view::copy(queue, hist_hbuf, hist_dbuf, 1u);
    alpaka::wait::wait(queue);
    assert(0 == hist->off[0]);
    assert(N == hist->size());

    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
            workDiv, bin(), alpaka::mem::view::getPtrNative(hist_dbuf), alpaka::mem::view::getPtrNative(v_dbuf), N));

    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDiv,
                                                                             forEachInWindow(),
                                                                             alpaka::mem::view::getPtrNative(hist_dbuf),
                                                                             alpaka::mem::view::getPtrNative(v_dbuf),
                                                                             N,
                                                                             NBINS,
                                                                             DELTA));

    alpaka::wait::wait(queue);
  }
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
