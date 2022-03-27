#include <iostream>

#include <hip/hip_runtime.h>

#include "CUDACore/cudaCheck.h"

__global__ void print() { printf("GPU thread %d\n", static_cast<int>(threadIdx.x)); }

int main() {
  std::cout << "World from" << std::endl;
  print<<<1, 4>>>();
  cudaCheck(hipDeviceSynchronize());
  return 0;
}
