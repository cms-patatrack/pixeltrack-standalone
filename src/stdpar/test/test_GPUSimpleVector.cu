//  author: Felice Pantaleo, CERN, 2018
#include <cassert>
#include <iostream>
#include <new>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDACore/SimpleVector.h"
#include "CUDACore/cudaCheck.h"

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

  auto maxN = 10000;

  auto obj_unique_ptr = std::make_unique<cms::cuda::SimpleVector<int>>();
  auto data_unique_ptr = std::make_unique<int[]>(maxN);
  auto obj_ptr = obj_unique_ptr.get();
  auto data_ptr = data_unique_ptr.get();


  auto v = cms::cuda::make_SimpleVector(obj_ptr, maxN, data_ptr);

  assert(obj_ptr->size() == 0);
  assert(obj_ptr->capacity() == static_cast<int>(maxN));

  int numBlocks = 5;
  int numThreadsPerBlock = 256;
  vector_pushback<<<numBlocks, numThreadsPerBlock>>>(obj_ptr);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));
  vector_reset<<<numBlocks, numThreadsPerBlock>>>(obj_ptr);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  assert(obj_ptr->size() == 0);

  vector_emplace_back<<<numBlocks, numThreadsPerBlock>>>(obj_ptr);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));

  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
