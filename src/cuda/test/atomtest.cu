#include <cassert>
#include <iostream>
#include <new>

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

using Data = float;

__global__ void shared_block(Data *d, int n) {
  __shared__ Data s;
  int block = blockIdx.x;
  if (threadIdx.x == 0) {
    s = 0.0;
  }
  __syncthreads();

  for (int i = 0; i < 200000; i++) {
    atomicAdd(&s, 1.0);
    atomicAdd(&s, -1.0);
  }
  atomicAdd(&s, 1.0);

  __syncthreads();

  if (threadIdx.x == 0) {
    d[block] = s;
  }
}

__global__ void global_block(Data *d, int n) {
  int block = blockIdx.x;
  for (int i = 0; i < 200000; i++) {
    atomicAdd(&d[block], 1.0);
    atomicAdd(&d[block], -1.0);
  }
  atomicAdd(&d[block], 1.0);
}

__global__ void shared_grid(Data *d, int n) {
  __shared__ Data var;
  if (threadIdx.x == 0) {
    var = 0.0;
  }

  __syncthreads();

  for (int i = 0; i < 200000; i++) {
    atomicAdd(&var, 1.0);
    atomicAdd(&var, -1.0);
  }
  atomicAdd(&var, 1.0);

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(&d[0], var);
  }
}

__global__ void global_grid(Data *d, int n) {
  for (int i = 0; i < 200000; i++) {
    atomicAdd(&d[0], 1.0);
    atomicAdd(&d[0], -1.0);
  }
  atomicAdd(&d[0], 1.0);
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

  // run version with static shared memory
  cudaMemcpy(d_d, &order, n * sizeof(Data), cudaMemcpyHostToDevice);
  gettimeofday(&t1, 0);
  shared_block<<<blocks, thr>>>(d_d, n);
  cudaDeviceSynchronize();
  gettimeofday(&t2, 0);
  double time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0;
  printf("Shared block:  %f s \n", time);
  cudaMemcpy(backup, d_d, n * sizeof(Data), cudaMemcpyDeviceToHost);
  for (int i = 0; i < blocks; i++) {
    if (backup[i] != (Data)thr)
      printf("Error: d[%d] != r (%f, %d)\n", i, backup[i], thr);
  }

  // run version with global memory
  cudaMemcpy(d_d, &order, n * sizeof(Data), cudaMemcpyHostToDevice);
  gettimeofday(&t1, 0);
  global_block<<<blocks, thr>>>(d_d, n);
  cudaDeviceSynchronize();
  gettimeofday(&t2, 0);
  time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0;
  printf("Global block:  %f s \n", time);
  cudaMemcpy(backup, d_d, n * sizeof(Data), cudaMemcpyDeviceToHost);
  for (int i = 0; i < blocks; i++) {
    if (backup[i] != (Data)thr)
      printf("Error: d[%d] !=r (%f, %d)\n", i, backup[i], thr);
  }

  // run version with shared memory
  cudaMemcpy(d_d, order, n * sizeof(Data), cudaMemcpyHostToDevice);
  gettimeofday(&t1, 0);
  shared_grid<<<blocks, thr>>>(d_d, n);
  cudaDeviceSynchronize();
  gettimeofday(&t2, 0);
  time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0;
  printf("Shared grid:  %f s \n", time);
  cudaMemcpy(backup, d_d, n * sizeof(Data), cudaMemcpyDeviceToHost);
  if (backup[0] != (Data)n)
    printf("Error: d !=r (%f, %d)\n", backup[0], n);

  // run version with global memory
  cudaMemcpy(d_d, order, n * sizeof(Data), cudaMemcpyHostToDevice);
  gettimeofday(&t1, 0);
  global_grid<<<blocks, thr>>>(d_d, n);
  cudaDeviceSynchronize();
  gettimeofday(&t2, 0);
  time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0 / 1000.0;
  printf("Global grid:  %f s \n", time);
  cudaMemcpy(backup, d_d, 1 * sizeof(Data), cudaMemcpyDeviceToHost);
  if (backup[0] != (Data)n)
    printf("Error: d !=r (%f, %d)\n", backup[0], n);
}