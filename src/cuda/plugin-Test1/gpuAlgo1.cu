#include "gpuAlgo1.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

namespace {
  constexpr int NUM_VALUES = 4000;

  template <typename T>
  __global__ void vectorAdd(const T *a, const T *b, T *c, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
      c[i] = a[i] + b[i];
    }
  }

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
}  // namespace

cms::cuda::device::unique_ptr<float[]> gpuAlgo1(cudaStream_t stream) {
  auto h_a = cms::cuda::make_host_unique<float[]>(NUM_VALUES, stream);
  auto h_b = cms::cuda::make_host_unique<float[]>(NUM_VALUES, stream);

  for (auto i = 0; i < NUM_VALUES; i++) {
    h_a[i] = i;
    h_b[i] = i * i;
  }

  auto d_a = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);
  auto d_b = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);

  cudaCheck(cudaMemcpyAsync(d_a.get(), h_a.get(), NUM_VALUES * sizeof(float), cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaMemcpyAsync(d_b.get(), h_b.get(), NUM_VALUES * sizeof(float), cudaMemcpyHostToDevice, stream));

  int threadsPerBlock{32};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  auto d_c = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);
  auto current_device = cms::cuda::currentDevice();
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
}
