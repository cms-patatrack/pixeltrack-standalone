#include <iostream>
#include <string>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename T, typename Data>
struct shared_block {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Data* vec, T elements) const {
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    auto blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
    Data b = 1.0;
    Data c = -1.0;

    auto& s = alpaka::declareSharedVar<Data, __COUNTER__>(acc);

    if (threadIdxLocal == 0) {
      s = 0;
    }

    syncBlockThreads(acc);

    for ([[maybe_unused]] T index :
         ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::elements_with_stride(acc, elements)) {
      for (int i = 0; i < 200000; i++) {
        alpaka::atomicAdd(acc, &s, b, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &s, c, alpaka::hierarchy::Blocks{});
      }
      alpaka::atomicAdd(acc, &s, b, alpaka::hierarchy::Blocks{});
    }

    syncBlockThreads(acc);

    if (threadIdxLocal == 0) {
      vec[blockIdxInGrid] = s;
    }
  }
};

template <typename T, typename Data>
struct global_block {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Data* vec, T elements) const {
    auto blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
    Data b = 1.0;
    Data c = -1.0;

    for ([[maybe_unused]] T index :
         ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::elements_with_stride(acc, elements)) {
      for (int i = 0; i < 200000; i++) {
        alpaka::atomicAdd(acc, &vec[blockIdxInGrid], b, alpaka::hierarchy::Grids{});
        alpaka::atomicAdd(acc, &vec[blockIdxInGrid], c, alpaka::hierarchy::Grids{});
      }
      alpaka::atomicAdd(acc, &vec[blockIdxInGrid], b, alpaka::hierarchy::Grids{});
    }
  }
};

template <typename T, typename Data>
struct global_grid {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Data* vec, T elements) const {
    Data b = 1.0;
    Data c = -1.0;

    for ([[maybe_unused]] T index :
         ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::elements_with_stride(acc, elements)) {
      for (int i = 0; i < 200000; i++) {
        alpaka::atomicAdd(acc, &vec[0], b, alpaka::hierarchy::Grids{});  //alpaka::hierarchy::Blocks/Threads/Grids
        alpaka::atomicAdd(acc, &vec[0], c, alpaka::hierarchy::Grids{});
      }
      alpaka::atomicAdd(acc, &vec[0], b, alpaka::hierarchy::Grids{});
    }
  }
};

template <typename T, typename Data>
struct shared_grid {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Data* vec, T elements) const {
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    Data b = 1.0;
    Data c = -1.0;

    auto& s = alpaka::declareSharedVar<Data, __COUNTER__>(acc);

    if (threadIdxLocal == 0) {
      s = 0;
    }

    syncBlockThreads(acc);

    for ([[maybe_unused]] T index :
         ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::elements_with_stride(acc, elements)) {
      for (int i = 0; i < 200000; i++) {
        alpaka::atomicAdd(acc, &s, b, alpaka::hierarchy::Blocks{});  //alpaka::hierarchy::Blocks/Threads/Grids
        alpaka::atomicAdd(acc, &s, c, alpaka::hierarchy::Blocks{});
      }
      alpaka::atomicAdd(acc, &s, b, alpaka::hierarchy::Blocks{});
    }

    syncBlockThreads(acc);

    if (threadIdxLocal == 0) {
      alpaka::atomicAdd(acc, &vec[0], s, alpaka::hierarchy::Grids{});
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
  auto workDivMultiBlockInit1 = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
      blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

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
                      workDivMultiBlockInit1, shared_block<Idx, Data>(), alpaka::getPtrNative(order), num_items));
  alpaka::wait(queue_1_0);
  auto endT = std::chrono::high_resolution_clock::now();
  std::cout << "Shared Block: " << std::chrono::duration<double>(endT - beginT).count() << " s" << std::endl;
  alpaka::memcpy(queue_1_0, res, order, num_items);
  for (Idx i = 0; i < nBlocksInit; i++) {
    if (res_ptr[i] != (Data)nThreadsInit)
      std::cout << "[" << i << "]:  " << res_ptr[i] << " != " << (Data)num_items << std::endl;
  }

  // Run on global memory
  alpaka::memcpy(queue_1_0, order, bufHostA, num_items);
  beginT = std::chrono::high_resolution_clock::now();
  alpaka::enqueue(queue_1_0,
                  alpaka::createTaskKernel<Acc1D>(
                      workDivMultiBlockInit1, global_block<Idx, Data>(), alpaka::getPtrNative(order), num_items));
  alpaka::wait(queue_1_0);
  endT = std::chrono::high_resolution_clock::now();
  std::cout << "Global Block: " << std::chrono::duration<double>(endT - beginT).count() << " s" << std::endl;
  alpaka::memcpy(queue_1_0, res, order, num_items);
  for (Idx i = 0; i < nBlocksInit; i++) {
    if (res_ptr[i] != (Data)nThreadsInit)
      std::cout << "[" << i << "]:  " << res_ptr[i] << " != " << (Data)num_items << std::endl;
  }

  // Run on Shared memory
  alpaka::memcpy(queue_1_0, order, bufHostA, num_items);
  beginT = std::chrono::high_resolution_clock::now();
  alpaka::enqueue(queue_1_0,
                  alpaka::createTaskKernel<Acc1D>(
                      workDivMultiBlockInit1, shared_grid<Idx, Data>(), alpaka::getPtrNative(order), num_items));
  alpaka::wait(queue_1_0);
  endT = std::chrono::high_resolution_clock::now();
  std::cout << "Shared Grid: " << std::chrono::duration<double>(endT - beginT).count() << " s" << std::endl;
  alpaka::memcpy(queue_1_0, res, order, num_items);
  if (res_ptr[0] != (Data)num_items)
    std::cout << "[0]:  " << res_ptr[0] << " != " << (Data)num_items << std::endl << std::endl;

  // Run on Global memory
  alpaka::memcpy(queue_1_0, order, bufHostA, num_items);
  beginT = std::chrono::high_resolution_clock::now();
  alpaka::enqueue(queue_1_0,
                  alpaka::createTaskKernel<Acc1D>(
                      workDivMultiBlockInit1, global_grid<Idx, Data>(), alpaka::getPtrNative(order), num_items));
  alpaka::wait(queue_1_0);
  endT = std::chrono::high_resolution_clock::now();
  std::cout << "Global Grid: " << std::chrono::duration<double>(endT - beginT).count() << " s" << std::endl;
  alpaka::memcpy(queue_1_0, res, order, num_items);
  if (res_ptr[0] != (Data)num_items)
    std::cout << "[0]:  " << res_ptr[0] << " != " << (Data)num_items << std::endl;

  return 0;
}
