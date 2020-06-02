#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cstdint>

using member_type = Kokkos::TeamPolicy<KokkosExecSpace>::member_type;
#ifdef TODO
template <typename T>
KOKKOS_FUNCTION warpPrefixScan(T const* __restrict__ ci, T* __restrict__ co, uint32_t i, uint32_t mask) {
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = threadIdx.x & 0x1f;

  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

#endif // TODO

//same as above may remove
template <typename T>
KOKKOS_FUNCTION warpPrefixScan(T* c, uint32_t i, uint32_t mask) {
  auto x = c[i];
  auto laneId = threadIdx.x & 0x1f;
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#ifdef TODO
// limited to 32*32 elements....
template <typename T>
__device__ __host__ void __forceinline__ blockPrefixScan(T const* __restrict__ ci,
                                                         T* __restrict__ co,
                                                         uint32_t size,
                                                         T* ws
#ifndef __CUDA_ARCH__
                                                         = nullptr
#endif
) {
#ifdef __CUDA_ARCH__
  assert(ws);
  assert(size <= 1024);
  assert(0 == blockDim.x % 32);
  auto first = threadIdx.x;
  auto mask = __ballot_sync(0xffffffff, first < size);

  for (auto i = first; i < size; i += blockDim.x) {
    warpPrefixScan(ci, co, i, mask);
    auto laneId = threadIdx.x & 0x1f;
    auto warpId = i / 32;
    assert(warpId < 32);
    if (31 == laneId)
      ws[warpId] = co[i];
    mask = __ballot_sync(mask, i + blockDim.x < size);
  }
  __syncthreads();
  if (size <= 32)
    return;
  if (threadIdx.x < 32)
    warpPrefixScan(ws, threadIdx.x, 0xffffffff);
  __syncthreads();
  for (auto i = first + 32; i < size; i += blockDim.x) {
    auto warpId = i / 32;
    co[i] += ws[warpId - 1];
  }
  __syncthreads();
#else
  co[0] = ci[0];
  for (uint32_t i = 1; i < size; ++i)
    co[i] = ci[i] + co[i - 1];
#endif
}

#endif // TODO

// same as above, may remove
// limited to 32*32 elements....
template <typename T>
KOKKOS_FUNCTION void blockPrefixScan(T* c,
                                     uint32_t size,
                                     T* ws,
                                     const member_type& teamMember
) {
  assert(ws);
  assert(size <= 1024);
  assert(0 == teamMember.team_size() % 32);
  auto first = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank();
  auto mask = __ballot_sync(0xffffffff, first < size);

  for (auto i = first; i < size; i += teamMember.team_size()) {
    warpPrefixScan(c, i, mask);
    auto laneId = first & 0x1f;
    auto warpId = i / 32;
    assert(warpId < 32);
    if (31 == laneId)
      ws[warpId] = c[i];
    mask = __ballot_sync(mask, i + teamMember.team_size() < size);
  }
  teamMember.team_barrier();
  if (size <= 32)
    return;
  if (first < 32)
    warpPrefixScan(ws, first, 0xffffffff);
  teamMember.team_barrier();
  for (auto i = first + 32; i < size; i += teamMember.team_size()) {
    auto warpId = i / 32;
    c[i] += ws[warpId - 1];
  }
  teamMember.team_barrier();

}

#ifdef TODO

// limited to 1024*1024 elements....
template <typename T>
__global__ void multiBlockPrefixScan(T const* __restrict__ ci, T* __restrict__ co, int32_t size, int32_t* pc) {
  __shared__ T ws[32];
  // first each block does a scan of size 1024; (better be enough blocks....)
  assert(1024 * gridDim.x >= size);
  int off = 1024 * blockIdx.x;
  if (size - off > 0)
    blockPrefixScan(ci + off, co + off, std::min(1024, size - off), ws);

  // count blocks that finished
  __shared__ bool isLastBlockDone;
  if (0 == threadIdx.x) {
    auto value = atomicAdd(pc, 1);  // block counter
    isLastBlockDone = (value == (int(gridDim.x) - 1));
  }

  __syncthreads();

  if (!isLastBlockDone)
    return;

  // good each block has done its work and now we are left in last block

  // let's get the partial sums from each block
  __shared__ T psum[1024];
  for (int i = threadIdx.x, ni = gridDim.x; i < ni; i += blockDim.x) {
    auto j = 1024 * i + 1023;
    psum[i] = (j < size) ? co[j] : T(0);
  }
  __syncthreads();
  blockPrefixScan(psum, psum, gridDim.x, ws);

  // now it would have been handy to have the other blocks around...
  int first = threadIdx.x;                                 // + blockDim.x * blockIdx.x
  for (int i = first + 1024; i < size; i += blockDim.x) {  //  *gridDim.x) {
    auto k = i / 1024;                                     // block
    co[i] += psum[k - 1];
  }
}

#endif // TODO

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
