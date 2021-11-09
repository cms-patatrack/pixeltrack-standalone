#include <cassert>
#include <iostream>
#include <new>

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

using Data = float;

__device__ void rand_func() {
  int sum = 0;
  for (int i = 0; i < 1000; i++)
    sum += i;
}

__device__ void iterate(int id) {
  for (int j = id; j < 10000; j++) {
    if (j % 2 == 0) {
      rand_func();
    }
  }
}

__global__ void check_sync(Data *d, int no_threads) {
  for (int i = 0; i < no_threads * no_threads; i++) {
    if ((i % no_threads) == threadIdx.x) {
      iterate(threadIdx.x);
    }
    __syncthreads();
  }
}

int main(void) {
  const int n = 1 << 10;

  const int thr = 1024;
  const int blocks = (n + thr - 1) / thr;

  printf("Threads/block:%d blocks/grid:%d\n", thr, blocks);

  Data *d_d;
  cudaMalloc(&d_d, n * sizeof(Data));
  struct timeval t1, t2;

  gettimeofday(&t1, 0);
  check_sync<<<blocks, thr>>>(d_d, n);
  cudaDeviceSynchronize();
  gettimeofday(&t2, 0);
  auto time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0;
  printf("Time: %f s\n", time);
}