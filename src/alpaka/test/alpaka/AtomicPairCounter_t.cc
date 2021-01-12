#include <cassert>
#include <iostream>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/AtomicPairCounter.h"


using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct update {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, cms::Alpaka::AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    // Global thread index in grid
    const uint32_t threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
    const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

    // Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
    const uint32_t firstElementIdxGlobal = threadIdxGlobal * threadDimension;
    const uint32_t endElementIdxGlobalUncut = firstElementIdxGlobal + threadDimension;
    const uint32_t endElementIdxGlobal = std::min(endElementIdxGlobalUncut, n);

for (uint32_t i = firstElementIdxGlobal; i < endElementIdxGlobal; ++i) {
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
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, cms::Alpaka::AtomicPairCounter const *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    assert(dc->get().m == n);
    ind[n] = dc->get().n;
  }
};

struct verify {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, cms::Alpaka::AtomicPairCounter const *dc, uint32_t const *ind, uint32_t const *cont, uint32_t n) const {
    // Global thread index in grid
    const uint32_t threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
    const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

    // Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
    const uint32_t firstElementIdxGlobal = threadIdxGlobal * threadDimension;
    const uint32_t endElementIdxGlobalUncut = firstElementIdxGlobal + threadDimension;
    const uint32_t endElementIdxGlobal = std::min(endElementIdxGlobalUncut, n);

    for (uint32_t i = firstElementIdxGlobal; i < endElementIdxGlobal; ++i) {
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
  const DevAcc device(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
  Queue queue(device);

  constexpr uint32_t C = 1;
  const Vec sizeC(C);
  auto c_dbuf = alpaka::mem::buf::alloc<cms::Alpaka::AtomicPairCounter, Idx>(device, sizeC);
  alpaka::mem::view::set(queue, c_dbuf, 0, sizeC);

  std::cout << "size " << C * sizeof(cms::Alpaka::AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  const Vec sizeN(N);
  const Vec sizeM(M);
  auto n_dbuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, sizeN);
  auto m_dbuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, sizeM);

  constexpr uint32_t NUM_VALUES = 10000;

  // Prepare 1D workDiv
  Vec elementsPerThread(Vec::all(1));
  Vec threadsPerBlock(Vec::all(512));
  const Vec blocksPerGrid(Vec::all(2000));
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED || \
  ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_BT_OMP4_ENABLED
  // on the GPU, run with 32 threads in parallel per block, each looking at a single element
  // on the CPU, run serially with a single thread per block, over 32 elements
  std::swap(threadsPerBlock, elementsPerThread);
#endif
  const WorkDiv workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

  alpaka::queue::enqueue(queue,
			 alpaka::kernel::createTaskKernel<Acc>(workDiv,
							       update(),
							       alpaka::mem::view::getPtrNative(c_dbuf),
							       alpaka::mem::view::getPtrNative(n_dbuf),
							       alpaka::mem::view::getPtrNative(m_dbuf),
							       NUM_VALUES));

  const Vec elementsPerThreadFinalize(Vec::all(1));
  const Vec threadsPerBlockFinalize(Vec::all(1));
  const Vec blocksPerGridFinalize(Vec::all(1));
  const WorkDiv workDivFinalize(blocksPerGridFinalize, threadsPerBlockFinalize, elementsPerThreadFinalize);
  alpaka::queue::enqueue(queue,
			 alpaka::kernel::createTaskKernel<Acc>(workDivFinalize,
							       finalize(),
							       alpaka::mem::view::getPtrNative(c_dbuf),
							       alpaka::mem::view::getPtrNative(n_dbuf),
							       alpaka::mem::view::getPtrNative(m_dbuf),
							       NUM_VALUES));
 
  alpaka::queue::enqueue(queue,
			 alpaka::kernel::createTaskKernel<Acc>(workDiv,
							       verify(),
							       alpaka::mem::view::getPtrNative(c_dbuf),
							       alpaka::mem::view::getPtrNative(n_dbuf),
							       alpaka::mem::view::getPtrNative(m_dbuf),
							       NUM_VALUES));
    
  auto c_hbuf = alpaka::mem::buf::alloc<cms::Alpaka::AtomicPairCounter, Idx>(host, sizeC);
  alpaka::mem::view::copy(queue, c_hbuf, c_dbuf, sizeC);
  alpaka::wait::wait(queue);  

  auto c_h = alpaka::mem::view::getPtrNative(c_hbuf);
  std::cout << c_h->get().n << ' ' << c_h->get().m << std::endl;

  return 0;
}
