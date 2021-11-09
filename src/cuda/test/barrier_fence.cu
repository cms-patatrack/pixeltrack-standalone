#include <cassert>
#include <iostream>
#include <new>

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>

using Data = float;

__global__ void global_thread(Data *d, int n) {
  int no_blocks = 128;

  for (int i = 0; i < no_blocks * no_blocks * 10; i++) {
    if (i % no_blocks == blockIdx.x) {
      if (i % no_blocks > 0) {
        d[blockIdx.x] = d[blockIdx.x - 1] + 1;
      }
    }
    __threadfence();
  }
}

__global__ void shared_fence(Data *d, int n) {
  __shared__ Data s[256];
  int block = blockIdx.x;
  int no_threads = 256;

  for (int i = 0; i < no_threads * no_threads * 10; i++) {
    if (i % no_threads == threadIdx.x && threadIdx.x > 0) {
      s[threadIdx.x] = s[threadIdx.x - 1] + 1;
    }
    __threadfence();
  }

  if (threadIdx.x == 0) {
    d[block] = s[127] + s[129];
  }
}

int main(void) {
  const int n = 1 << 15;

  const int thr = 256;
  const int blocks = (n + thr - 1) / thr;
  Data order[n], backup[n];

  printf("Threads/block:%d blocks/grid:%d\n", thr, blocks);

  for (int i = 0; i < n; i++)
    order[i] = 0.0;

  Data *d_d;
  cudaMalloc(&d_d, n * sizeof(Data));
  struct timeval t1, t2;

  // run version with shared memory
  cudaMemcpy(d_d, order, n * sizeof(Data), cudaMemcpyHostToDevice);
  gettimeofday(&t1, 0);
  shared_fence<<<blocks, thr>>>(d_d, n);
  cudaDeviceSynchronize();
  gettimeofday(&t2, 0);
  auto time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0;
  printf("Shared time: %f\n", time);
  cudaMemcpy(backup, d_d, n * sizeof(Data), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 128; i++) {
    if (backup[i] != 256.0)
      printf("Error1: d[%d] != r (%f, %d)\n", i, backup[i], n / 128);
  }

  // run version with global memory
  cudaMemcpy(d_d, order, n * sizeof(Data), cudaMemcpyHostToDevice);
  gettimeofday(&t1, 0);
  global_thread<<<blocks, thr>>>(d_d, n);
  cudaDeviceSynchronize();
  gettimeofday(&t2, 0);
  time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0;
  printf("Global time: %f\n", time);
  cudaMemcpy(backup, d_d, n * sizeof(Data), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 128; i++) {
    if (backup[i] != Data(i))
      printf("Error1: d[%d] != r (%f, %d)\n", i, backup[i], i);
  }
}