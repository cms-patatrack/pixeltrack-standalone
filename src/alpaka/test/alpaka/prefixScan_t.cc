#include <iostream>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/prefixScan.h"

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
    auto& ws = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
    auto& c = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);
    auto& co = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::for_each_element_in_block_strided(
        acc, size, [&](uint32_t i) { c[i] = 1; });

    alpaka::syncBlockThreads(acc);

    ::cms::alpakatools::blockPrefixScan(acc, c, co, size, ws);
    ::cms::alpakatools::blockPrefixScan(acc, c, size, ws);

    assert(1 == c[0]);
    assert(1 == co[0]);

    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::for_each_element_in_block_strided(acc, size, 1u, [&](uint32_t i) {
      assert(c[i] == c[i - 1] + 1);
      assert(c[i] == i + 1);
      assert(c[i] == co[i]);
    });
  }
};

/*
 * NB: GPU-only, so do not care about elements here.
 */
template <typename T>
struct testWarpPrefixScan {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t size) const {
    assert(size <= 32);
    auto& c = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);
    auto& co = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

    uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
    uint32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    auto i = blockThreadIdx;
    c[i] = 1;
    alpaka::syncBlockThreads(acc);
    auto laneId = blockThreadIdx & 0x1f;

    warpPrefixScan(laneId, c, co, i, 0xffffffff);
    warpPrefixScan(laneId, c, i, 0xffffffff);

    alpaka::syncBlockThreads(acc);

    assert(1 == c[0]);
    assert(1 == co[0]);
    if (i != 0) {
      if (c[i] != c[i - 1] + 1)
        printf(format_traits<T>::failed_msg, size, i, blockDimension, c[i], c[i - 1]);
      assert(c[i] == c[i - 1] + 1);
      assert(c[i] == i + 1);
      assert(c[i] == co[i]);
    }
  }
};

struct init {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t* v, uint32_t val, uint32_t n) const {
    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::for_each_element_in_grid(acc, n, [&](uint32_t index) {
      v[index] = val;

      if (index == 0)
        printf("init\n");
    });
  }
};

struct verify {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t const* v, uint32_t n) const {
    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::for_each_element_in_grid(acc, n, [&](uint32_t index) {
      assert(v[index] == index + 1);

      if (index == 0)
        printf("verify\n");
    });
  }
};

int main() {
  const DevHost host(alpaka::getDevByIdx<PltfHost>(0u));
  const Device device(alpaka::getDevByIdx<Platform>(0u));
  const Vec1D size(1u);

  Queue queue(device);

  // WARP PREFIXSCAN (OBVIOUSLY GPU-ONLY)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  std::cout << "warp level" << std::endl;

  const Vec1D threadsPerBlockOrElementsPerThread1(Vec1D::all(32));
  const Vec1D blocksPerGrid1(Vec1D::all(1));
  const WorkDiv1D& workDivWarp = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
      blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 32));

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 16));

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 5));
#endif

  // PORTABLE BLOCK PREFIXSCAN
  std::cout << "block level" << std::endl;

  // Running kernel with 1 block, and bs threads per block or elements per thread.
  // NB: obviously for tests only, for perf would need to use bs = 1024 in GPU version.
  for (int bs = 32; bs <= 1024; bs += 32) {
    const Vec1D threadsPerBlockOrElementsPerThread2(Vec1D::all(bs));
    const Vec1D blocksPerGrid2(Vec1D::all(1));
    const WorkDiv1D& workDivSingleBlock = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
        blocksPerGrid2, threadsPerBlockOrElementsPerThread2);

    std::cout << "blocks per grid: " << blocksPerGrid2
              << ", threads per block or elements per thread: " << threadsPerBlockOrElementsPerThread2 << std::endl;

    // Problem size
    for (int j = 1; j <= 1024; ++j) {
      alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivSingleBlock, testPrefixScan<uint16_t>(), j));
      alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivSingleBlock, testPrefixScan<float>(), j));
    }
  }

  // PORTABLE MULTI-BLOCK PREFIXSCAN
  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    std::cout << "multiblock" << std::endl;
    num_items *= 10;

    auto input_dBuf = alpaka::allocBuf<uint32_t, Idx>(device, Vec1D::all(num_items));
    uint32_t* input_d = alpaka::getPtrNative(input_dBuf);

    auto output1_dBuf = alpaka::allocBuf<uint32_t, Idx>(device, Vec1D::all(num_items));
    uint32_t* output1_d = alpaka::getPtrNative(output1_dBuf);

    const auto nThreadsInit = 256;  // NB: 1024 would be better
    // Just kept here to be identical to CUDA test
    const Vec1D threadsPerBlockOrElementsPerThread3(Vec1D::all(nThreadsInit));
    const auto nBlocksInit = (num_items + nThreadsInit - 1) / nThreadsInit;
    const Vec1D blocksPerGrid3(Vec1D::all(nBlocksInit));
    const WorkDiv1D& workDivMultiBlockInit = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
        blocksPerGrid3, threadsPerBlockOrElementsPerThread3);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivMultiBlockInit, init(), input_d, 1, num_items));

    const auto nThreads = 1024;
    const Vec1D threadsPerBlockOrElementsPerThread4(Vec1D::all(nThreads));
    const auto nBlocks = (num_items + nThreads - 1) / nThreads;
    const Vec1D blocksPerGrid4(Vec1D::all(nBlocks));
    const WorkDiv1D& workDivMultiBlock = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
        blocksPerGrid4, threadsPerBlockOrElementsPerThread4);

    std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nBlocks << std::endl;
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDivMultiBlock,
                                                    ::cms::alpakatools::multiBlockPrefixScanFirstStep<uint32_t>(),
                                                    input_d,
                                                    output1_d,
                                                    num_items));

    const Vec1D blocksPerGridSecondStep(Vec1D::all(1));
    const WorkDiv1D& workDivMultiBlockSecondStep = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
        blocksPerGridSecondStep, threadsPerBlockOrElementsPerThread4);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDivMultiBlockSecondStep,
                                                    ::cms::alpakatools::multiBlockPrefixScanSecondStep<uint32_t>(),
                                                    input_d,
                                                    output1_d,
                                                    num_items,
                                                    nBlocks));

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivMultiBlock, verify(), output1_d, num_items));

    alpaka::wait(queue);  // input_dBuf and output1_dBuf end of scope
  }                       // ksize

  return 0;
}
