#include <iostream>
#include <string>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/threadfence.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename T, typename Data>
struct global_fence {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Data* vec, T elements) const {
    auto blockIdxLocal(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
    int no_blocks = 128;

    for (int i = 0; i < no_blocks * no_blocks * 10; i++) {
      if (i % no_blocks == (int)blockIdxLocal) {
        if (i % no_blocks > 0) {
          vec[blockIdxLocal] = vec[blockIdxLocal - 1] + 1;
        }
      }
      cms::alpakatools::threadfence(acc);
    }
  }
};

template <typename T, typename Data>
struct shared_fence {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Data* vec, T elements) const {
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    auto blockIdxLocal(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

    auto& s = alpaka::declareSharedVar<Data[256], __COUNTER__>(acc);

    for (int i = 0; i < 256 * 256 * 10; i++) {
      if (i % 256 == (int)threadIdxLocal && threadIdxLocal > 0) {
        s[threadIdxLocal] = s[threadIdxLocal - 1] + 1;
      }
      cms::alpakatools::threadfence(acc);
    }

    if (threadIdxLocal == 0) {
      vec[blockIdxLocal] = s[127] + s[129];
    }
  }
};

int main(void) {
  using Dim = alpaka::DimInt<1u>;
  using Data = float;
  const Idx num_items = 1 << 15;
  Idx nThreadsInit = 256;
  Idx nBlocksInit = (num_items + nThreadsInit - 1) / nThreadsInit;

  const Device device_1(alpaka::getDevByIdx<Platform>(0u));
  alpaka::Queue<Device, alpaka::Blocking> queue_1_0(device_1);
  alpaka::Queue<Device, alpaka::Blocking> queue_1_1(device_1);

  const Vec1D threadsPerBlockOrElementsPerThread1(Vec1D::all(nThreadsInit));
  const Vec1D blocksPerGrid1(Vec1D::all(nBlocksInit));
  auto workDivMultiBlockInit1 = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

  using DevHost = alpaka::DevCpu;
  auto const devHost = alpaka::getDevByIdx<DevHost>(0u);

  using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
  BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devHost, num_items));
  BufHost res(alpaka::allocBuf<Data, Idx>(devHost, num_items));

  Data* const pBufHostA(alpaka::getPtrNative(bufHostA));
  Data* const res_ptr(alpaka::getPtrNative(res));

  for (Idx i = 0; i < num_items; i++) {
    pBufHostA[i] = 0.0;
  }

  using BufAcc = alpaka::Buf<Device, Data, Dim, Idx>;
  BufAcc order(alpaka::allocBuf<Data, Idx>(device_1, num_items));

  printf("Threads/block:%d blocks/grid:%d\n", threadsPerBlockOrElementsPerThread1[0u], blocksPerGrid1[0u]);

  // Run on shared memory
  alpaka::memcpy(queue_1_0, order, bufHostA, num_items);
  auto beginT = std::chrono::high_resolution_clock::now();
  alpaka::enqueue(queue_1_0,
                  alpaka::createTaskKernel<Acc1D>(
                      workDivMultiBlockInit1, shared_fence<Idx, Data>(), alpaka::getPtrNative(order), num_items));
  alpaka::wait(queue_1_0);
  auto endT = std::chrono::high_resolution_clock::now();
  std::cout << "Shared time: " << std::chrono::duration<double>(endT - beginT).count() << " s" << std::endl;
  alpaka::memcpy(queue_1_0, res, order, num_items);
  for (int i = 0; i < 128; i++) {
    if (res_ptr[i] != 256.0)
      printf("Error: d[%d] != r (%f, %d)\n", i, res_ptr[i], i);
  }

  // Run on global memory
  alpaka::memcpy(queue_1_0, order, bufHostA, num_items);
  beginT = std::chrono::high_resolution_clock::now();
  alpaka::enqueue(queue_1_0,
                  alpaka::createTaskKernel<Acc1D>(
                      workDivMultiBlockInit1, global_fence<Idx, Data>(), alpaka::getPtrNative(order), num_items));
  alpaka::wait(queue_1_0);
  endT = std::chrono::high_resolution_clock::now();
  std::cout << "Global time: " << std::chrono::duration<double>(endT - beginT).count() << " s" << std::endl;
  alpaka::memcpy(queue_1_0, res, order, num_items);
  for (int i = 0; i < 128; i++) {
    if (res_ptr[i] != Data(i))
      printf("Error: d[%d] != r (%f, %d)\n", i, res_ptr[i], i);
  }

  return 0;
}
