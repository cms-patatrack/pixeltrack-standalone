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
        cms::alpakatools::element_global_index_range_truncated(acc, Vec1::all(n));

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
        cms::alpakatools::element_global_index_range_truncated(acc, Vec1::all(n));

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


  // WARP PREFIXSCAN (OBVIOUSLY GPU-ONLY)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  std::cout << "warp level" << std::endl;

  const Vec1 threadsPerBlockOrElementsPerThread1(Vec1::all(32));
  const Vec1 blocksPerGrid1(Vec1::all(1));
  const WorkDiv1 &workDivWarp = cms::alpakatools::make_workdiv(blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

  alpaka::queue::enqueue(queue, alpaka::kernel::createTaskKernel<Acc1>(workDivWarp, testWarpPrefixScan<int>(), 32));
  alpaka::wait::wait(queue);

  alpaka::queue::enqueue(queue, alpaka::kernel::createTaskKernel<Acc1>(workDivWarp, testWarpPrefixScan<int>(), 16));
  alpaka::wait::wait(queue);

  alpaka::queue::enqueue(queue, alpaka::kernel::createTaskKernel<Acc1>(workDivWarp, testWarpPrefixScan<int>(), 5));
  alpaka::wait::wait(queue);
#endif

  // PORTABLE BLOCK PREFIXSCAN
  std::cout << "block level" << std::endl;
  // running kernel with 1 block, and bs threads per block or elements per thread

  int bs = 1;
  for (bs = 32; bs <= 1024; bs += 32) {
  
  const Vec1 threadsPerBlockOrElementsPerThread2(Vec1::all(bs));
  const Vec1 blocksPerGrid2(Vec1::all(1));
  const WorkDiv1 &workDivSingleBlock = cms::alpakatools::make_workdiv(blocksPerGrid2, threadsPerBlockOrElementsPerThread2);

  std::cout << "blocks per grid: " << blocksPerGrid2 
	    << ", threads per block or elements per thread: " << threadsPerBlockOrElementsPerThread2
            << std::endl;

    for (int j = 1; j <= 1024; ++j) {
      alpaka::queue::enqueue(queue,
                             alpaka::kernel::createTaskKernel<Acc1>(workDivSingleBlock, testPrefixScan<uint16_t>(), j));
      alpaka::wait::wait(queue);
      alpaka::queue::enqueue(queue,
                             alpaka::kernel::createTaskKernel<Acc1>(workDivSingleBlock, testPrefixScan<float>(), j));
      alpaka::wait::wait(queue);
    }
  }

  alpaka::wait::wait(queue);

  // PORTABLE MULTI-BLOCK PREFIXSCAN
  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {  
    std::cout << "multiblock" << std::endl;
    num_items *= 10;

    auto input_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, Vec1::all(num_items * sizeof(uint32_t)));
    uint32_t* input_d = alpaka::mem::view::getPtrNative(input_dBuf);

    auto output1_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, Vec1::all(num_items * sizeof(uint32_t)));
    uint32_t* output1_d = alpaka::mem::view::getPtrNative(output1_dBuf);

    const auto nThreadsInit = 256;
    const Vec1 threadsPerBlockOrElementsPerThread3(Vec1::all(nThreadsInit));
    const auto nBlocksInit = (num_items + nThreadsInit - 1) / nThreadsInit;
    const Vec1 blocksPerGrid3(Vec1::all(nBlocksInit));
    const WorkDiv1 &workDivMultiBlockInit = cms::alpakatools::make_workdiv(blocksPerGrid3, threadsPerBlockOrElementsPerThread3);

    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc1>(workDivMultiBlockInit, init(), input_d, 1, num_items));
    alpaka::wait::wait(queue);


    const auto nThreads = 1024;
    const Vec1 threadsPerBlockOrElementsPerThread4(Vec1::all(nThreads));
    const auto nBlocks = (num_items + nThreads - 1) / nThreads;
    const Vec1 blocksPerGrid4(Vec1::all(nBlocks));
    const WorkDiv1 &workDivMultiBlock = cms::alpakatools::make_workdiv(blocksPerGrid4, threadsPerBlockOrElementsPerThread4);

    std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nBlocks << std::endl;
    alpaka::queue::enqueue(
			   queue,
			   alpaka::kernel::createTaskKernel<Acc1>(workDivMultiBlock,
								  multiBlockPrefixScanFirstStep<uint32_t>(),
								  input_d,
								  output1_d,
								  num_items));
    alpaka::wait::wait(queue);

    const Vec1 blocksPerGridSecondStep(Vec1::all(1));
    const WorkDiv1 &workDivMultiBlockSecondStep = cms::alpakatools::make_workdiv(blocksPerGridSecondStep, threadsPerBlockOrElementsPerThread4);
    alpaka::queue::enqueue(
			   queue,
			   alpaka::kernel::createTaskKernel<Acc1>(workDivMultiBlockSecondStep,
								  multiBlockPrefixScanSecondStep<uint32_t>(),
								  input_d,
								  output1_d,
								  num_items,
								  nBlocks));
    alpaka::wait::wait(queue);

    alpaka::queue::enqueue(
			   queue,
			   alpaka::kernel::createTaskKernel<Acc1>(workDivMultiBlock, 
								  verify(), output1_d, num_items));
    alpaka::wait::wait(queue);

  }  // ksize
  return 0;
}
