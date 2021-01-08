#include "hip/hip_runtime.h"
#include <iostream>

#include "CUDACore/cudaCheck.h"
#include "CUDACore/prefixScan.h"
#include "CUDACore/requireDevices.h"

using namespace cms::hip;

template <typename T>
struct format_traits {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %d %d\n";
};

template <>
struct format_traits<float> {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %f %f\n";
};

template <typename T>
__global__ void testPrefixScan(uint32_t size) {
  __shared__ T ws[warpSize];
  __shared__ T c[1024];
  __shared__ T co[1024];

  int first = threadIdx.x;
  for (uint32_t i = first; i < size; i += static_cast<uint32_t>(blockDim.x))
    c[i] = 1;
  __syncthreads();

  blockPrefixScan(c, co, size, ws);
  blockPrefixScan(c, size, ws);

  assert(1 == c[0]);
  assert(1 == co[0]);
  for (uint32_t i = first + 1; i < size; i += static_cast<uint32_t>(blockDim.x)) {
    if (c[i] != c[i - 1] + 1)
      printf(format_traits<T>::failed_msg, size, i, static_cast<int>(blockDim.x), c[i], c[i - 1]);
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] = co[i]);
  }
}

template <typename T>
__global__ void testWarpPrefixScan(uint32_t size) {
  assert(size <= warpSize);
  __shared__ T c[1024];
  __shared__ T co[1024];
  int i = threadIdx.x;
  c[i] = 1;
  __syncthreads();

  warpPrefixScan(c, co, i);
  warpPrefixScan(c, i);
  __syncthreads();

  assert(1 == c[0]);
  assert(1 == co[0]);
  if (i != 0) {
    if (c[i] != c[i - 1] + 1)
      printf(format_traits<T>::failed_msg, size, i, static_cast<int>(blockDim.x), c[i], c[i - 1]);
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] = co[i]);
  }
}

__global__ void init(uint32_t *v, uint32_t val, uint32_t n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    v[i] = val;
  if (i == 0)
    printf("init\n");
}

__global__ void verify(uint32_t const *v, uint32_t n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    assert(v[i] == i + 1);
  if (i == 0)
    printf("verify\n");
}

int main() {
  cms::hiptest::requireDevices();

  std::cout << "warp level" << std::endl;
  if (warpSize > 32) {
    // std::cout << "warp 64" << std::endl;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(testWarpPrefixScan<int>), dim3(1), dim3(warpSize), 0, 0, 64);
    hipDeviceSynchronize();
  }
  // std::cout << "warp 32" << std::endl;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(testWarpPrefixScan<int>), dim3(1), dim3(warpSize), 0, 0, 32);
  hipDeviceSynchronize();
  // std::cout << "warp 16" << std::endl;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(testWarpPrefixScan<int>), dim3(1), dim3(warpSize), 0, 0, 16);
  hipDeviceSynchronize();
  // std::cout << "warp 5" << std::endl;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(testWarpPrefixScan<int>), dim3(1), dim3(warpSize), 0, 0, 5);
  hipDeviceSynchronize();

  std::cout << "block level" << std::endl;
  for (int bs = warpSize; bs <= 1024; bs += warpSize) {
    // std::cout << "bs " << bs << std::endl;
    for (int j = 1; j <= 1024; ++j) {
      // std::cout << j << std::endl;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(testPrefixScan<uint16_t>), dim3(1), dim3(bs), 0, 0, j);
      hipDeviceSynchronize();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(testPrefixScan<float>), dim3(1), dim3(bs), 0, 0, j);
      hipDeviceSynchronize();
    }
  }
  hipDeviceSynchronize();

  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    std::cout << "multiblok" << std::endl;
    // Declare, allocate, and initialize device-accessible pointers for input and output
    num_items *= 10;
    uint32_t *d_in;
    uint32_t *d_out1;
    uint32_t *d_out2;

    cudaCheck(hipMalloc(&d_in, num_items * sizeof(uint32_t)));
    cudaCheck(hipMalloc(&d_out1, num_items * sizeof(uint32_t)));
    cudaCheck(hipMalloc(&d_out2, num_items * sizeof(uint32_t)));

    auto nthreads = 256;
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    hipLaunchKernelGGL(init, dim3(nblocks), dim3(nthreads), 0, 0, d_in, 1, num_items);

    // the block counter
    int32_t *d_pc;
    cudaCheck(hipMalloc(&d_pc, sizeof(int32_t)));
    cudaCheck(hipMemset(d_pc, 0, sizeof(int32_t)));

    nthreads = 1024;
    nblocks = (num_items + nthreads - 1) / nthreads;
    std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nblocks << std::endl;
    hipLaunchKernelGGL(multiBlockPrefixScan, dim3(nblocks), dim3(nthreads), 4 * nblocks, 0, d_in, d_out1, num_items, d_pc);
    cudaCheck(hipGetLastError());
    hipLaunchKernelGGL(verify, dim3(nblocks), dim3(nthreads), 0, 0, d_out1, num_items);
    cudaCheck(hipGetLastError());
    hipDeviceSynchronize();

  }  // ksize
  return 0;
}
