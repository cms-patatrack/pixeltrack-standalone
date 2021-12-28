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
    cms::alpakatools::for_each_element_in_grid(acc, n, [&](uint32_t i) {
      auto m = i % 11;
      m = m % 6 + 1;  // max 6, no 0
      auto c = dc->add(acc, m);
      assert(c.m < n);
      ind[c.m] = c.n;
      for (uint32_t j = c.n; j < c.n + m; ++j)
        cont[j] = i;
    });
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
    cms::alpakatools::for_each_element_in_grid(acc, n, [&](uint32_t i) {
      assert(0 == ind[0]);
      assert(dc->get().m == n);
      assert(ind[n] == dc->get().n);
      auto ib = ind[i];
      auto ie = ind[i + 1];
#ifndef NDEBUG
      auto k = cont[ib++];
#endif
      assert(k < n);
      for (; ib < ie; ++ib)
        assert(cont[ib] == k);
    });
  }
};

int main() {
  const DevHost host(alpaka::getDevByIdx<PltfHost>(0u));
  const Device device(alpaka::getDevByIdx<Platform>(0u));
  Queue queue(device);

  constexpr uint32_t C = 1;
  const Vec1D sizeC(C);
  auto c_dbuf = alpaka::allocBuf<cms::alpakatools::AtomicPairCounter, Idx>(device, sizeC);
  alpaka::memset(queue, c_dbuf, 0);

  std::cout << "size " << C * sizeof(cms::alpakatools::AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  const Vec1D sizeN(N);
  const Vec1D sizeM(M);
  auto n_dbuf = alpaka::allocBuf<uint32_t, Idx>(device, sizeN);
  auto m_dbuf = alpaka::allocBuf<uint32_t, Idx>(device, sizeM);

  constexpr uint32_t NUM_VALUES = 10000;

  // Update
  const Vec1D &blocksPerGrid(Vec1D(2000u));
  const Vec1D &threadsPerBlockOrElementsPerThread(Vec1D(512u));
  const WorkDiv1D &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(workDiv,
                                                  update(),
                                                  alpaka::getPtrNative(c_dbuf),
                                                  alpaka::getPtrNative(n_dbuf),
                                                  alpaka::getPtrNative(m_dbuf),
                                                  NUM_VALUES));

  // Finalize
  const Vec1D &blocksPerGridFinalize(Vec1D(1u));
  const Vec1D &threadsPerBlockOrElementsPerThreadFinalize(Vec1D(1u));
  const WorkDiv1D &workDivFinalize =
      cms::alpakatools::make_workdiv(blocksPerGridFinalize, threadsPerBlockOrElementsPerThreadFinalize);
  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(workDivFinalize,
                                                  finalize(),
                                                  alpaka::getPtrNative(c_dbuf),
                                                  alpaka::getPtrNative(n_dbuf),
                                                  alpaka::getPtrNative(m_dbuf),
                                                  NUM_VALUES));

  // Verify
  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(workDiv,
                                                  verify(),
                                                  alpaka::getPtrNative(c_dbuf),
                                                  alpaka::getPtrNative(n_dbuf),
                                                  alpaka::getPtrNative(m_dbuf),
                                                  NUM_VALUES));

  auto c_hbuf = alpaka::allocBuf<cms::alpakatools::AtomicPairCounter, Idx>(host, sizeC);
  alpaka::memcpy(queue, c_hbuf, c_dbuf);
  alpaka::wait(queue);

  auto c_h = alpaka::getPtrNative(c_hbuf);
  std::cout << c_h->get().n << ' ' << c_h->get().m << std::endl;

  return 0;
}
