#include <cassert>
#include <iostream>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/AtomicPairCounter.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct update {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(
      const T_Acc &acc, cms::alpakatools::AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    const auto &[firstElementIdxGlobal, endElementIdxGlobal] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(n));

    for (uint32_t i = firstElementIdxGlobal[0u]; i < endElementIdxGlobal[0u]; ++i) {
      auto m = i % 11;
      m = m % 6 + 1;  // max 6, no 0
      auto c = dc->add(acc, m);
      assert(c.m < n);
      ind[c.m] = c.n;
      for (uint32_t j = c.n; j < c.n + m; ++j)
        cont[j] = i;
    }
  }
};

struct finalize {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                cms::alpakatools::AtomicPairCounter const *dc,
                                uint32_t *ind,
                                uint32_t *cont,
                                uint32_t n) const {
    assert(dc->get().m == n);
    ind[n] = dc->get().n;
  }
};

struct verify {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                cms::alpakatools::AtomicPairCounter const *dc,
                                uint32_t const *ind,
                                uint32_t const *cont,
                                uint32_t n) const {
    const auto &[firstElementIdxGlobal, endElementIdxGlobal] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(n));

    for (uint32_t i = firstElementIdxGlobal[0u]; i < endElementIdxGlobal[0u]; ++i) {
      assert(0 == ind[0]);
      assert(dc->get().m == n);
      assert(ind[n] == dc->get().n);
      auto ib = ind[i];
      auto ie = ind[i + 1];
      auto k = cont[ib++];
      assert(k < n);
      for (; ib < ie; ++ib)
        assert(cont[ib] == k);
    }
  }
};

int main() {
  const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
  const DevAcc1 device(alpaka::pltf::getDevByIdx<PltfAcc1>(0u));
  Queue queue(device);

  constexpr uint32_t C = 1;
  const Vec1 sizeC(C);
  auto c_dbuf = alpaka::mem::buf::alloc<cms::alpakatools::AtomicPairCounter, Idx>(device, sizeC);
  alpaka::mem::view::set(queue, c_dbuf, 0, sizeC);

  std::cout << "size " << C * sizeof(cms::alpakatools::AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  const Vec1 sizeN(N);
  const Vec1 sizeM(M);
  auto n_dbuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, sizeN);
  auto m_dbuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, sizeM);

  constexpr uint32_t NUM_VALUES = 10000;

  // Update
  const Vec1 &blocksPerGrid(Vec1(2000u));
  const Vec1 &threadsPerBlockOrElementsPerThread(Vec1(512u));
  const WorkDiv1 &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
  alpaka::queue::enqueue(queue,
                         alpaka::kernel::createTaskKernel<Acc1>(workDiv,
                                                                update(),
                                                                alpaka::mem::view::getPtrNative(c_dbuf),
                                                                alpaka::mem::view::getPtrNative(n_dbuf),
                                                                alpaka::mem::view::getPtrNative(m_dbuf),
                                                                NUM_VALUES));

  // Finalize
  const Vec1 &blocksPerGridFinalize(Vec1(1u));
  const Vec1 &threadsPerBlockOrElementsPerThreadFinalize(Vec1(1u));
  const WorkDiv1 &workDivFinalize =
      cms::alpakatools::make_workdiv(blocksPerGridFinalize, threadsPerBlockOrElementsPerThreadFinalize);
  alpaka::queue::enqueue(queue,
                         alpaka::kernel::createTaskKernel<Acc1>(workDivFinalize,
                                                                finalize(),
                                                                alpaka::mem::view::getPtrNative(c_dbuf),
                                                                alpaka::mem::view::getPtrNative(n_dbuf),
                                                                alpaka::mem::view::getPtrNative(m_dbuf),
                                                                NUM_VALUES));

  // Verify
  alpaka::queue::enqueue(queue,
                         alpaka::kernel::createTaskKernel<Acc1>(workDiv,
                                                                verify(),
                                                                alpaka::mem::view::getPtrNative(c_dbuf),
                                                                alpaka::mem::view::getPtrNative(n_dbuf),
                                                                alpaka::mem::view::getPtrNative(m_dbuf),
                                                                NUM_VALUES));

  auto c_hbuf = alpaka::mem::buf::alloc<cms::alpakatools::AtomicPairCounter, Idx>(host, sizeC);
  alpaka::mem::view::copy(queue, c_hbuf, c_dbuf, sizeC);
  alpaka::wait::wait(queue);

  auto c_h = alpaka::mem::view::getPtrNative(c_hbuf);
  std::cout << c_h->get().n << ' ' << c_h->get().m << std::endl;

  return 0;
}
