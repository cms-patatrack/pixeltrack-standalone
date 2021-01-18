#include <iostream>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/prefixScan.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename T>
struct format_traits {
public:
  static const constexpr char* failed_msg = "failed %d %d %d: %d %d\n";
};

template <>
struct format_traits<float> {
public:
  static const constexpr char* failed_msg = "failed %d %d %d: %f %f\n";
};

template <typename T>
struct testPrefixScan {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, unsigned int size) const {
    auto&& ws = alpaka::block::shared::st::allocVar<T[32], __COUNTER__>(acc);
    auto&& c = alpaka::block::shared::st::allocVar<T[1024], __COUNTER__>(acc);
    auto&& co = alpaka::block::shared::st::allocVar<T[1024], __COUNTER__>(acc);

    uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
    uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

    auto first = blockThreadIdx;
    for (auto i = first; i < size; i += blockDimension)
      c[i] = 1;
    alpaka::block::sync::syncBlockThreads(acc);

    blockPrefixScan(acc, c, co, size, ws);
    blockPrefixScan(acc, c, size, ws);

    assert(1 == c[0]);
    assert(1 == co[0]);
    for (auto i = first + 1; i < size; i += blockDimension) {
      assert(c[i] == c[i - 1] + 1);
      assert(c[i] == i + 1);
      assert(c[i] = co[i]);
    }
  }
};

template <typename T>
struct testWarpPrefixScan {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t size) const {
    assert(size <= 32);
    auto&& c = alpaka::block::shared::st::allocVar<T[1024], __COUNTER__>(acc);
    auto&& co = alpaka::block::shared::st::allocVar<T[1024], __COUNTER__>(acc);

    uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
    uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    auto i = blockThreadIdx;
    c[i] = 1;
    alpaka::block::sync::syncBlockThreads(acc);
    auto laneId = blockThreadIdx & 0x1f;

    warpPrefixScan(laneId, c, co, i, 0xffffffff);
    warpPrefixScan(laneId, c, i, 0xffffffff);

    alpaka::block::sync::syncBlockThreads(acc);

    assert(1 == c[0]);
    assert(1 == co[0]);
    if (i != 0) {
      if (c[i] != c[i - 1] + 1)
        printf(format_traits<T>::failed_msg, size, i, blockDimension, c[i], c[i - 1]);
      assert(c[i] == c[i - 1] + 1);
      assert(c[i] == i + 1);
      assert(c[i] = co[i]);
    }
  }
};

struct init {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t* v, uint32_t val, uint32_t n) const {
    const auto& [firstElementIdxGlobal, endElementIdxGlobal] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(n));

    for (uint32_t index = firstElementIdxGlobal[0u]; index < endElementIdxGlobal[0u]; ++index) {
      v[index] = val;

      if (index == 0)
        printf("init\n");
    }
  }
};

struct verify {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t const* v, uint32_t n) const {
    const auto& [firstElementIdxGlobal, endElementIdxGlobal] =
        cms::alpakatools::element_global_index_range(acc, Vec1::all(n));

    for (uint32_t index = firstElementIdxGlobal[0u]; index < endElementIdxGlobal[0u]; ++index) {
      assert(v[index] == index + 1);
      if (index == 0)
        printf("verify\n");
    }
  }
};

int main() {
  const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
  const DevAcc1 device(alpaka::pltf::getDevByIdx<PltfAcc1>(0u));
  const Vec1 size(1u);

  Queue queue(device);

  Vec1 elementsPerThread(Vec1::all(1));
  Vec1 threadsPerBlock(Vec1::all(32));
  Vec1 blocksPerGrid(Vec1::all(1));
#if defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_BT_OMP4_ENABLED
  // on the GPU, run with 512 threads in parallel per block, each looking at a single element
  // on the CPU, run serially with a single thread per block, over 512 elements
  std::swap(threadsPerBlock, elementsPerThread);
#endif
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  threadsPerBlock = Vec1::all(1);
#endif

  const WorkDiv1 workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);
  std::cout << "blocks per grid: " << blocksPerGrid << ", threads per block: " << threadsPerBlock
            << ", elements per thread: " << elementsPerThread << std::endl;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  std::cout << "warp level" << std::endl;
  alpaka::queue::enqueue(queue, alpaka::kernel::createTaskKernel<Acc1>(workDiv, testWarpPrefixScan<int>(), 32));
  alpaka::wait::wait(queue);

  alpaka::queue::enqueue(queue, alpaka::kernel::createTaskKernel<Acc1>(workDiv, testWarpPrefixScan<int>(), 16));
  alpaka::wait::wait(queue);

  alpaka::queue::enqueue(queue, alpaka::kernel::createTaskKernel<Acc1>(workDiv, testWarpPrefixScan<int>(), 5));
  alpaka::wait::wait(queue);
#endif
  std::cout << "block level" << std::endl;
  int bs = 1;
#if not defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

  for (bs = 32; bs <= 1024; bs += 32) {
#endif
    std::cout << "bs " << bs << std::endl;
    for (int j = 1; j <= 1024; ++j) {
      // running kernel with 1 block, bs threads per block, 1 element per thread
      alpaka::queue::enqueue(queue,
                             alpaka::kernel::createTaskKernel<Acc1>(
                                 WorkDiv1{Vec1::all(1), Vec1::all(bs), Vec1::all(1)}, testPrefixScan<uint16_t>(), j));
      alpaka::wait::wait(queue);
      alpaka::queue::enqueue(queue,
                             alpaka::kernel::createTaskKernel<Acc1>(
                                 WorkDiv1{Vec1::all(1), Vec1::all(bs), Vec1::all(1)}, testPrefixScan<float>(), j));
      alpaka::wait::wait(queue);
    }
#if not defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  }
#endif

  alpaka::wait::wait(queue);

  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    std::cout << "multiblock" << std::endl;
    num_items *= 8;
    uint32_t* d_in;
    uint32_t* d_out1;
    uint32_t* d_out2;

    auto input_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, Vec1::all(num_items * sizeof(uint32_t)));
    uint32_t* input_d = alpaka::mem::view::getPtrNative(input_dBuf);

    auto output1_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, Vec1::all(num_items * sizeof(uint32_t)));
    uint32_t* output1_d = alpaka::mem::view::getPtrNative(output1_dBuf);

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    auto nthreads = 1;
#else
    auto nthreads = 256;
#endif
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc1>(
            WorkDiv1{Vec1::all(nblocks), Vec1::all(nthreads), Vec1::all(1)}, init(), input_d, 1, num_items));
    alpaka::wait::wait(queue);

    auto psum_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, Vec1::all(num_items * sizeof(uint32_t)));
    uint32_t* psum_d = alpaka::mem::view::getPtrNative(psum_dBuf);

    alpaka::mem::view::set(queue, psum_dBuf, 0u, Vec1::all(num_items * sizeof(uint32_t)));

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    nthreads = 1;
    auto nelements = 768;
    nblocks = (num_items + nelements - 1) / nelements;
#else
    nthreads = 768;
    auto nelements = 1;
    nblocks = (num_items + nthreads - 1) / nthreads;
#endif

    std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nblocks << std::endl;
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc1>(WorkDiv1{Vec1::all(nblocks), Vec1::all(nthreads), Vec1::all(nelements)},
                                               multiBlockPrefixScanFirstStep<uint32_t>(),
                                               input_d,
                                               output1_d,
                                               psum_d,
                                               num_items));
    alpaka::wait::wait(queue);
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc1>(WorkDiv1{Vec1::all(1), Vec1::all(nthreads), Vec1::all(nelements)},
                                               multiBlockPrefixScanSecondStep<uint32_t>(),
                                               input_d,
                                               output1_d,
                                               psum_d,
                                               num_items,
                                               nblocks));
    alpaka::wait::wait(queue);

    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc1>(
            WorkDiv1{Vec1::all(nblocks), Vec1::all(nthreads), Vec1::all(nelements)}, verify(), output1_d, num_items));
    alpaka::wait::wait(queue);

  }  // ksize
  return 0;
}
