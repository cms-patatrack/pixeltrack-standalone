#include "hip/hip_runtime.h"
#include <iostream>

#include <hip/hip_runtime.h>

#include "CUDACore/cudaCheck.h"

__global__ void print() { printf("GPU thread %d\n", threadIdx.x); }

int main() {
  std::cout << "World from" << std::endl;
  hipLaunchKernelGGL(print, dim3(1), dim3(4), 0, 0);
  cudaCheck(hipDeviceSynchronize());
  return 0;
}
