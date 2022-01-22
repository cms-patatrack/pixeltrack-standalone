#include <cassert>
#include <iostream>

#include "AlpakaCore/AtomicPairCounter.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemoryHelper.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct update {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      const TAcc &acc, AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    for_each_element_in_grid(acc, n, [&](uint32_t i) {
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
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      const TAcc &acc, AtomicPairCounter const *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    assert(dc->get().m == n);
    ind[n] = dc->get().n;
  }
};

struct verify {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      const TAcc &acc, AtomicPairCounter const *dc, uint32_t const *ind, uint32_t const *cont, uint32_t n) const {
    for_each_element_in_grid(acc, n, [&](uint32_t i) {
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

  auto c_d = make_device_buffer<AtomicPairCounter>(queue);
  alpaka::memset(queue, c_d, 0);

  std::cout << "size " << sizeof(AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  auto n_d = make_device_buffer<uint32_t[]>(queue, N);
  auto m_d = make_device_buffer<uint32_t[]>(queue, M);

  constexpr uint32_t NUM_VALUES = 10000;

  // Update
  const auto blocksPerGrid = 2000u;
  const auto threadsPerBlockOrElementsPerThread = 512u;
  const auto workDiv = make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(workDiv, update(), c_d.data(), n_d.data(), m_d.data(), NUM_VALUES));

  // Finalize
  const auto blocksPerGridFinalize = 1u;
  const auto threadsPerBlockOrElementsPerThreadFinalize = 1u;
  const auto workDivFinalize = make_workdiv(blocksPerGridFinalize, threadsPerBlockOrElementsPerThreadFinalize);
  alpaka::enqueue(
      queue,
      alpaka::createTaskKernel<Acc1D>(workDivFinalize, finalize(), c_d.data(), n_d.data(), m_d.data(), NUM_VALUES));

  // Verify
  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(workDiv, verify(), c_d.data(), n_d.data(), m_d.data(), NUM_VALUES));

  auto c_h = make_host_buffer<AtomicPairCounter>();
  alpaka::memcpy(queue, c_h, c_d);
  alpaka::wait(queue);

  std::cout << c_h->get().n << ' ' << c_h->get().m << std::endl;

  return 0;
}
