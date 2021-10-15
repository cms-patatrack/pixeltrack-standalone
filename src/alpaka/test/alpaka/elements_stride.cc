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
    for (T index : ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::elements_with_stride(acc, elements)) {
      printf("range:%d\n", index);
    }
  }
};

template <typename T>
struct explicit_loop_1d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3D elements) const {
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
    auto blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));

    auto blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
    auto gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc));

    Vec3D thread({blockDimension[0u] * blockIdxInGrid[0u] + threadIdxLocal[0u], Idx{0}, Idx{0}});
    Vec3D stride({blockDimension[0u] * gridDimension[0u], Idx{1}, Idx{1}});
    Vec3D index(thread);

    for (T i = index[0u]; i < elements[0u]; i += stride[0u]) {
      printf("explicit_loop_1d:%d:%d:%d\n", i, index[1u], index[2u]);
    }
  }
};

template <typename T>
struct range_loop_1d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3D elements) const {
    for (Vec3D index : ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::elements_with_stride_1d(acc, elements)) {
      printf("range_1d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
    }
  }
};

template <typename T>
struct explicit_loop_2d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3D elements) const {
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
    auto blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));

    auto blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
    auto gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc));

    Vec3D thread({blockDimension[0u] * blockIdxInGrid[0u] + threadIdxLocal[0u],
                  blockDimension[1u] * blockIdxInGrid[1u] + threadIdxLocal[1u],
                  Idx{0}});
    Vec3D stride({blockDimension[0u] * gridDimension[0u], blockDimension[1u] * gridDimension[1u], Idx{1}});
    Vec3D index(thread);

    for (T i = index[1u]; i < elements[1u]; i += stride[1u]) {
      for (T i = index[0u]; i < elements[0u]; i += stride[0u]) {
        printf("explicit_loop_2d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
      }
    }
  }
};

template <typename T>
struct range_loop_2d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3D elements) const {
    for (Vec3D index : ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::elements_with_stride_2d(acc, elements)) {
      printf("range_2d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
    }
  }
};

template <typename T>
struct explicit_loop_3d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3D elements) const {
    auto threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
    auto blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));

    auto blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
    auto gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc));

    Vec3D thread({blockDimension[0u] * blockIdxInGrid[0u] + threadIdxLocal[0u],
                  blockDimension[1u] * blockIdxInGrid[1u] + threadIdxLocal[1u],
                  blockDimension[2u] * blockIdxInGrid[2u] + threadIdxLocal[2u]});
    Vec3D stride({blockDimension[0u] * gridDimension[0u],
                  blockDimension[1u] * gridDimension[1u],
                  blockDimension[2u] * gridDimension[2u]});
    Vec3D index(thread);

    for (T i = index[2u]; i < elements[2u]; i += stride[2u]) {
      for (T i = index[1u]; i < elements[1u]; i += stride[1u]) {
        for (T i = index[0u]; i < elements[0u]; i += stride[0u]) {
          printf("explicit_loop_3d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
        }
      }
    }
  }
};

template <typename T>
struct range_loop_3d {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, Vec3D elements) const {
    for (Vec3D index : ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::elements_with_stride_3d(acc, elements)) {
      printf("range_3d:%d:%d:%d\n", index[0u], index[1u], index[2u]);
    }
  }
};

int main(void) {
  const Idx num_items = 10;
  Idx nThreadsInit = 7;
  Vec3D elements(Vec3D::all(num_items));
  Idx nBlocksInit = (num_items + nThreadsInit - 1) / nThreadsInit;

  /*
  // Case for elements_with_stride (only one dimension)
  const Device device_1(alpaka::getDevByIdx<Platform>(0u));

  alpaka::Queue<Device, alpaka::Blocking> queue_1_0(device_1);
  alpaka::Queue<Device, alpaka::Blocking> queue_1_1(device_1);

  const Vec1D threadsPerBlockOrElementsPerThread1(Vec1D::all(nThreadsInit));
  const Vec1D blocksPerGrid1(Vec1D::all(nBlocksInit));
  auto workDivMultiBlockInit1 =
      ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

  printf("threads/block = %d and blocks/grid = %d\n", threadsPerBlockOrElementsPerThread1[0u], blocksPerGrid1[0u]);

  alpaka::enqueue(queue_1_0, alpaka::createTaskKernel<Acc1D>(workDivMultiBlockInit1, 
                  explicit_loop<Idx>(), num_items));

  alpaka::enqueue(queue_1_1, alpaka::createTaskKernel<Acc1D>(workDivMultiBlockInit1, 
                  range_loop<Idx>(), num_items));
  
  */
  // Case for elements_with_stride_Xd (3 dimensions)
  const Device device3(alpaka::getDevByIdx<Platform>(0u));

  alpaka::Queue<Device, alpaka::Blocking> queue_3_0(device3);
  alpaka::Queue<Device, alpaka::Blocking> queue_3_1(device3);

  const Vec3D threadsPerBlockOrElementsPerThread3(Vec3D::all(nThreadsInit));
  const Vec3D blocksPerGrid3(Vec3D::all(nBlocksInit));
  auto workDivMultiBlockInit3 = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
      blocksPerGrid3, threadsPerBlockOrElementsPerThread3);

  printf("threads/block = %d and blocks/grid = %d\n", threadsPerBlockOrElementsPerThread3[0u], blocksPerGrid3[0u]);
  alpaka::enqueue(queue_3_0,
                  alpaka::createTaskKernel<Acc3D>(workDivMultiBlockInit3, explicit_loop_1d<Idx>(), elements));
  alpaka::enqueue(queue_3_1, alpaka::createTaskKernel<Acc3D>(workDivMultiBlockInit3, range_loop_1d<Idx>(), elements));

  return 0;
}
