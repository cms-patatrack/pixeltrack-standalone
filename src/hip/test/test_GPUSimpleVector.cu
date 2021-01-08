#include "hip/hip_runtime.h"
//  author: Felice Pantaleo, CERN, 2018
#include <cassert>
#include <iostream>
#include <new>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

#include "CUDACore/SimpleVector.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/requireDevices.h"

__global__ void vector_pushback(cms::cuda::SimpleVector<int> *foo) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  foo->push_back(index);
}

__global__ void vector_reset(cms::cuda::SimpleVector<int> *foo) { foo->reset(); }

__global__ void vector_emplace_back(cms::cuda::SimpleVector<int> *foo) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  foo->emplace_back(index);
}

int main() {
  cms::cudatest::requireDevices();

  auto maxN = 10000;
  cms::cuda::SimpleVector<int> *obj_ptr = nullptr;
  cms::cuda::SimpleVector<int> *d_obj_ptr = nullptr;
  cms::cuda::SimpleVector<int> *tmp_obj_ptr = nullptr;
  int *data_ptr = nullptr;
  int *d_data_ptr = nullptr;

  cudaCheck(hipHostMalloc(&obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  cudaCheck(hipHostMalloc(&data_ptr, maxN * sizeof(int)));
  cudaCheck(hipMalloc(&d_data_ptr, maxN * sizeof(int)));

  auto v = cms::cuda::make_SimpleVector(obj_ptr, maxN, data_ptr);

  cudaCheck(hipHostMalloc(&tmp_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  cms::cuda::make_SimpleVector(tmp_obj_ptr, maxN, d_data_ptr);
  assert(tmp_obj_ptr->size() == 0);
  assert(tmp_obj_ptr->capacity() == static_cast<int>(maxN));

  cudaCheck(hipMalloc(&d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  // ... and copy the object to the device.
  cudaCheck(hipMemcpy(d_obj_ptr, tmp_obj_ptr, sizeof(cms::cuda::SimpleVector<int>), hipMemcpyDefault));

  int numBlocks = 5;
  int numThreadsPerBlock = 256;
  hipLaunchKernelGGL(vector_pushback, dim3(numBlocks), dim3(numThreadsPerBlock), 0, 0, d_obj_ptr);
  cudaCheck(hipGetLastError());
  cudaCheck(hipDeviceSynchronize());

  cudaCheck(hipMemcpy(obj_ptr, d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>), hipMemcpyDefault));

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));
  hipLaunchKernelGGL(vector_reset, dim3(numBlocks), dim3(numThreadsPerBlock), 0, 0, d_obj_ptr);
  cudaCheck(hipGetLastError());
  cudaCheck(hipDeviceSynchronize());

  cudaCheck(hipMemcpy(obj_ptr, d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>), hipMemcpyDefault));

  assert(obj_ptr->size() == 0);

  hipLaunchKernelGGL(vector_emplace_back, dim3(numBlocks), dim3(numThreadsPerBlock), 0, 0, d_obj_ptr);
  cudaCheck(hipGetLastError());
  cudaCheck(hipDeviceSynchronize());

  cudaCheck(hipMemcpy(obj_ptr, d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>), hipMemcpyDefault));

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));

  cudaCheck(hipMemcpy(data_ptr, d_data_ptr, obj_ptr->size() * sizeof(int), hipMemcpyDefault));
  cudaCheck(hipHostFree(obj_ptr));
  cudaCheck(hipHostFree(data_ptr));
  cudaCheck(hipHostFree(tmp_obj_ptr));
  cudaCheck(hipFree(d_data_ptr));
  cudaCheck(hipFree(d_obj_ptr));
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
