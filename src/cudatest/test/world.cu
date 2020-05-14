#include <iostream>

#include <cuda_runtime.h>

#include "CUDACore/cudaCheck.h"

__global__ void print() { printf("GPU thread %d\n", threadIdx.x); }

int main() {
  std::cout << "World from" << std::endl;
  print<<<1, 4>>>();
  cudaCheck(cudaDeviceSynchronize());
  return 0;
}
