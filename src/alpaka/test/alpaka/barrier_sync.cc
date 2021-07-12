#include <iostream>
#include <string>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename T, typename Data>
struct check_sync {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Data* vec, T elements) const {
    int n = (int)elements;

    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    for (int i = 0; i < n * n; i++) {
      if (i % n == (int)threadIdxLocal) {
        for (int j = i; j < 10000; j++) {
          if (j % 2 == 0) {
            // Do random stuff
            int sum = 0;
            for (int k = 0; k < 1000; k++)
              sum += k;
          }
        }
      }
      syncBlockThreads(acc);
    }
  }
};

int main(void) {
  using Dim = alpaka::DimInt<1u>;
  using Data = float;
  const Idx num_items = 1 << 10;
  Idx nThreadsInit = 1024;
  Idx nBlocksInit = (num_items + nThreadsInit - 1) / nThreadsInit;

  const Device device_1(alpaka::getDevByIdx<Platform>(0u));
  alpaka::Queue<Device, alpaka::Blocking> queue_1_0(device_1);
  alpaka::Queue<Device, alpaka::Blocking> queue_1_1(device_1);

  const Vec1D threadsPerBlockOrElementsPerThread1(Vec1D::all(nThreadsInit));
  const Vec1D blocksPerGrid1(Vec1D::all(nBlocksInit));
  auto workDivMultiBlockInit1 = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

  using BufAcc = alpaka::Buf<Device, Data, Dim, Idx>;
  BufAcc order(alpaka::allocBuf<Data, Idx>(device_1, num_items));

  printf("Threads/block:%d blocks/grid:%d\n", threadsPerBlockOrElementsPerThread1[0u], blocksPerGrid1[0u]);

  // Run function
  auto beginT = std::chrono::high_resolution_clock::now();
  alpaka::enqueue(queue_1_0,
                  alpaka::createTaskKernel<Acc1D>(
                      workDivMultiBlockInit1, check_sync<Idx, Data>(), alpaka::getPtrNative(order), nThreadsInit));
  alpaka::wait(queue_1_0);
  auto endT = std::chrono::high_resolution_clock::now();
  std::cout << "Time: " << std::chrono::duration<double>(endT - beginT).count() << " s" << std::endl;

  return 0;
}
