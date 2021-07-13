#include <iostream>
#include <string>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename T>
struct explicit_loop {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, T elements) const {
    
    const T threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    const T blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

    const T blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
    const T gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u]);

    T thread = threadIdxLocal + blockDimension * blockIdxInGrid;
    T stride = blockDimension * gridDimension;

    for (T index(thread); index < elements; index += stride) {
        printf("explicit_loop:%d\n", index);
    }
  }
};

template <typename T>
struct range_loop {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, T elements) const {
    
    for (T index: cms::alpakatools::elements_with_stride<T, T_Acc>(acc, elements)) {
        printf("range:%d\n", index);
    }
  }
};

template <typename T>
struct explicit_loop_1d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3 elements) const {
    
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
    auto blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));

    auto blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
    auto gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc));

    Vec3 thread({blockDimension[0u] * blockIdxInGrid[0u] + threadIdxLocal[0u], 0, 0});
    Vec3 stride({blockDimension[0u] * gridDimension[0u], 1, 1});
    Vec3 index(thread);

    for (T i = index[0u]; i < elements[0u]; i += stride[0u]) {
      printf("explicit_loop_1d:%d:%d:%d\n", i, index[1u], index[2u]);
    }
    
  }
};


template <typename T>
struct range_loop_1d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3 elements) const {
    
    for (Vec3 index : cms::alpakatools::elements_with_stride_1d<T, T_Acc>(acc, elements)) {
      printf("range_1d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
    } 
  }
};

template <typename T>
struct explicit_loop_2d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3 elements) const {
    
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
    auto blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));

    auto blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
    auto gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc));

    Vec3 thread({blockDimension[0u] * blockIdxInGrid[0u] + threadIdxLocal[0u], blockDimension[1u] * blockIdxInGrid[1u] + threadIdxLocal[1u], 0});
    Vec3 stride({blockDimension[0u] * gridDimension[0u], blockDimension[1u] * gridDimension[1u], 1});
    Vec3 index(thread);

    for (T i = index[1u]; i < elements[1u]; i += stride[1u]) {
      for (T i = index[0u]; i < elements[0u]; i += stride[0u] ) {
        printf("explicit_loop_2d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
      }
    }
  }
};


template <typename T>
struct range_loop_2d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3 elements) const {
    
    for (Vec3 index : cms::alpakatools::elements_with_stride_2d<T, T_Acc>(acc, elements)) {
      printf("range_2d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
    } 
  }
};

template <typename T>
struct explicit_loop_3d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3 elements) const {
    
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
    auto blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));

    auto blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
    auto gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc));

    Vec3 thread({blockDimension[0u] * blockIdxInGrid[0u] + threadIdxLocal[0u],
                blockDimension[1u] * blockIdxInGrid[1u] + threadIdxLocal[1u],
                blockDimension[2u] * blockIdxInGrid[2u] + threadIdxLocal[2u]});
    Vec3 stride({blockDimension[0u] * gridDimension[0u],
                 blockDimension[1u] * gridDimension[1u],
                 blockDimension[2u] * gridDimension[2u]});
    Vec3 index(thread);

    for (T i = index[2u]; i < elements[2u]; i += stride[2u]) {
      for (T i = index[1u]; i < elements[1u]; i += stride[1u]) {
        for (T i = index[0u]; i < elements[0u]; i += stride[0u] ) {
          printf("explicit_loop_3d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
        }
      }
    }
  }
};


template <typename T>
struct range_loop_3d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3 elements) const {
    
    for (Vec3 index : cms::alpakatools::elements_with_stride_3d<T, T_Acc>(acc, elements)) {
      printf("range_3d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
    } 
  }
};


int main(void) {

  const Idx num_items = 10;
  Idx nThreadsInit = 7;
  Vec3 elements(Vec3::all(num_items));
  Idx nBlocksInit = (num_items + nThreadsInit - 1) / nThreadsInit;
  
  /*
  // Case for elements_with_stride (only one dimension)
  const DevAcc1 device_1(alpaka::getDevByIdx<PltfAcc1>(0u));

  alpaka::Queue<DevAcc1, alpaka::Blocking> queue_1_0(device_1);
  alpaka::Queue<DevAcc1, alpaka::Blocking> queue_1_1(device_1);

  const Vec1 threadsPerBlockOrElementsPerThread1(Vec1::all(nThreadsInit));
  const Vec1 blocksPerGrid1(Vec1::all(nBlocksInit));
  auto workDivMultiBlockInit1 =
      cms::alpakatools::make_workdiv(blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

  printf("threads/block = %d and blocks/grid = %d\n", threadsPerBlockOrElementsPerThread1[0u], blocksPerGrid1[0u]);

  alpaka::enqueue(queue_1_0, alpaka::createTaskKernel<Acc1>(workDivMultiBlockInit1, 
                  explicit_loop<Idx>(), num_items));

  alpaka::enqueue(queue_1_1, alpaka::createTaskKernel<Acc1>(workDivMultiBlockInit1, 
                  range_loop<Idx>(), num_items));
  
  */
  // Case for elements_with_stride_Xd (3 dimensions)
  const DevAcc3 device3(alpaka::getDevByIdx<PltfAcc1>(0u));

  alpaka::Queue<DevAcc3, alpaka::Blocking> queue_3_0(device3);
  alpaka::Queue<DevAcc3, alpaka::Blocking> queue_3_1(device3);
    
  const Vec3 threadsPerBlockOrElementsPerThread3(Vec3::all(nThreadsInit));
  const Vec3 blocksPerGrid3(Vec3::all(nBlocksInit));
  auto workDivMultiBlockInit3 =
        cms::alpakatools::make_workdiv(blocksPerGrid3, threadsPerBlockOrElementsPerThread3);

  printf("threads/block = %d and blocks/grid = %d\n", threadsPerBlockOrElementsPerThread3[0u], blocksPerGrid3[0u]);
      
    
  alpaka::enqueue(queue_3_0, alpaka::createTaskKernel<Acc3>(workDivMultiBlockInit3, 
                  explicit_loop_1d<Idx>(), elements));

  alpaka::enqueue(queue_3_1, alpaka::createTaskKernel<Acc3>(workDivMultiBlockInit3, 
                  range_loop_1d<Idx>(), elements));
  
  return 0;
}
