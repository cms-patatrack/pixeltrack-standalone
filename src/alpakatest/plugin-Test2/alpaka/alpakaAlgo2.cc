#include "alpakaAlgo2.h"

namespace {
  constexpr unsigned int NUM_VALUES = 1000;

  struct vectorAdd {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(
        T_Acc const &acc, T_Data const *a, T_Data const *b, T_Data *c, unsigned int numElements) const {
      //uint32_t const gridDimension(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const elemDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

      uint32_t i = (blockThreadIdx + gridBlockIdx * blockDimension) * elemDimension;
      if (i < numElements) {
        c[i] = a[i] + b[i];
      }
    }
  };

#ifdef TODO
  template <typename T>
  __global__ void vectorProd(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numElements && col < numElements) {
      c[row * numElements + col] = a[row] * b[col];
    }
  }

  template <typename T>
  __global__ void matrixMul(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numElements && col < numElements) {
      T tmp = 0;
      for (int i = 0; i < numElements; ++i) {
        tmp += a[row * numElements + i] * b[i * numElements + col];
      }
      c[row * numElements + col] = tmp;
    }
  }

  template <typename T>
  __global__ void matrixMulVector(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < numElements) {
      T tmp = 0;
      for (int i = 0; i < numElements; ++i) {
        tmp += a[row * numElements + i] * b[i];
      }
      c[row] = tmp;
    }
  }
#endif
}  // namespace

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  void alpakaAlgo2() {
    const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    const DevAcc device(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    const Vec size(NUM_VALUES);

    Queue queue(device);

    auto h_a_buf = alpaka::mem::buf::alloc<float, Idx>(host, size);
    auto h_b_buf = alpaka::mem::buf::alloc<float, Idx>(host, size);
    auto h_a = alpaka::mem::view::getPtrNative(h_a_buf);
    auto h_b = alpaka::mem::view::getPtrNative(h_b_buf);

    for (auto i = 0U; i < NUM_VALUES; i++) {
      h_a[i] = i;
      h_b[i] = i * i;
    }

    auto d_a_buf = alpaka::mem::buf::alloc<float, Idx>(device, size);
    auto d_b_buf = alpaka::mem::buf::alloc<float, Idx>(device, size);

    alpaka::mem::view::copy(queue, d_a_buf, h_a_buf, size);
    alpaka::mem::view::copy(queue, d_b_buf, h_b_buf, size);

    Vec elementsPerThread(Vec::all(1));
    Vec threadsPerBlock(Vec::all(32));
    Vec blocksPerGrid(Vec::all((NUM_VALUES + 32 - 1) / 32));
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED || \
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_BT_OMP4_ENABLED
    // on the GPU, run with 32 threads in parallel per block, each looking at a single element
    // on the CPU, run serially with a single thread per block, over 32 elements
    std::swap(threadsPerBlock, elementsPerThread);
#endif

    const WorkDiv workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

    auto d_c_buf = alpaka::mem::buf::alloc<float, Idx>(device, size);

    alpaka::queue::enqueue(queue,
                           alpaka::kernel::createTaskKernel<Acc>(workDiv,
                                                                 vectorAdd(),
                                                                 alpaka::mem::view::getPtrNative(d_a_buf),
                                                                 alpaka::mem::view::getPtrNative(d_b_buf),
                                                                 alpaka::mem::view::getPtrNative(d_c_buf),
                                                                 NUM_VALUES));

#ifdef TODO
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);

    auto d_ma = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
    auto d_mb = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
    auto d_mc = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
    dim3 threadsPerBlock3{NUM_VALUES, NUM_VALUES};
    dim3 blocksPerGrid3{1, 1};
    if (NUM_VALUES * NUM_VALUES > 32) {
      threadsPerBlock3.x = 32;
      threadsPerBlock3.y = 32;
      blocksPerGrid3.x = ceil(double(NUM_VALUES) / double(threadsPerBlock3.x));
      blocksPerGrid3.y = ceil(double(NUM_VALUES) / double(threadsPerBlock3.y));
    }
    vectorProd<<<blocksPerGrid3, threadsPerBlock3, 0, stream>>>(d_a.get(), d_b.get(), d_ma.get(), NUM_VALUES);
    vectorProd<<<blocksPerGrid3, threadsPerBlock3, 0, stream>>>(d_a.get(), d_c.get(), d_mb.get(), NUM_VALUES);
    matrixMul<<<blocksPerGrid3, threadsPerBlock3, 0, stream>>>(d_ma.get(), d_mb.get(), d_mc.get(), NUM_VALUES);

    matrixMulVector<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_mc.get(), d_b.get(), d_c.get(), NUM_VALUES);

    return d_a;
#endif

    alpaka::wait::wait(queue);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
