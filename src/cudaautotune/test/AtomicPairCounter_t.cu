#include <iostream>

#include <cuda_runtime.h>

#include "CUDACore/AtomicPairCounter.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/cuda_assert.h"

__global__ void update(cms::cuda::AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  auto m = i % 11;
  m = m % 6 + 1;  // max 6, no 0
  auto c = dc->add(m);
  assert(c.m < n);
  ind[c.m] = c.n;
  for (int j = c.n; j < c.n + m; ++j)
    cont[j] = i;
};

__global__ void finalize(cms::cuda::AtomicPairCounter const *dc, uint32_t *ind, uint32_t *cont, uint32_t n) {
  assert(dc->get().m == n);
  ind[n] = dc->get().n;
}

__global__ void verify(cms::cuda::AtomicPairCounter const *dc, uint32_t const *ind, uint32_t const *cont, uint32_t n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  assert(0 == ind[0]);
  assert(dc->get().m == n);
  assert(ind[n] == dc->get().n);
  auto ib = ind[i];
  auto ie = ind[i + 1];
  auto k = cont[ib++];
  assert(k < n);
  for (; ib < ie; ++ib)
    assert(cont[ib] == k);
}

int main() {
  cms::cuda::AtomicPairCounter *dc_d;
  cudaCheck(cudaMalloc(&dc_d, sizeof(cms::cuda::AtomicPairCounter)));
  cudaCheck(cudaMemset(dc_d, 0, sizeof(cms::cuda::AtomicPairCounter)));

  std::cout << "size " << sizeof(cms::cuda::AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  uint32_t *n_d, *m_d;
  cudaCheck(cudaMalloc(&n_d, N * sizeof(int)));
  // cudaMemset(n_d, 0, N*sizeof(int));
  cudaCheck(cudaMalloc(&m_d, M * sizeof(int)));

  update<<<2000, 512>>>(dc_d, n_d, m_d, 10000);
  finalize<<<1, 1>>>(dc_d, n_d, m_d, 10000);
  verify<<<2000, 512>>>(dc_d, n_d, m_d, 10000);

  cms::cuda::AtomicPairCounter dc;
  cudaCheck(cudaMemcpy(&dc, dc_d, sizeof(cms::cuda::AtomicPairCounter), cudaMemcpyDeviceToHost));

  std::cout << dc.get().n << ' ' << dc.get().m << std::endl;

  return 0;
}
