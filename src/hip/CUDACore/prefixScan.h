#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cstdint>

#include "hip/hip_runtime.h"

#include "CUDACore/cudaCompat.h"
#include "CUDACore/cuda_assert.h"

#ifdef __HIPCC__

template <typename T>
__device__ void __forceinline__ warpPrefixScan(T const* __restrict__ ci, T* __restrict__ co, uint32_t i) {
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = threadIdx.x & (warpSize-1);
#pragma unroll
  for (int offset = 1; offset < warpSize; offset <<= 1) {
    auto y = __shfl_up(x, offset);
    if (static_cast<int>(laneId) >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
__device__ void __forceinline__ warpPrefixScan(T* c, uint32_t i) {
  auto x = c[i];
  auto laneId = threadIdx.x & (warpSize-1);
#pragma unroll
  for (int offset = 1; offset < warpSize; offset <<= 1) {
    auto y = __shfl_up(x, offset);
    if (static_cast<int>(laneId) >= offset)
      x += y;
  }
  c[i] = x;
}

#endif // __HIPCC__

namespace cms {
  namespace hip {

    // limited to 32*32 elements....
    template <typename VT, typename T>
    __host__ __device__ __forceinline__ void blockPrefixScan(VT const* ci,
                                                             VT* co,
                                                             uint32_t size,
                                                             T* ws
#ifndef __HIP_DEVICE_COMPILE__
                                                             = nullptr
#endif
    ) {
#ifdef __HIP_DEVICE_COMPILE__
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDim.x % warpSize);
      uint32_t first = threadIdx.x;

      for (auto i = first; i < size; i += static_cast<uint32_t>(blockDim.x)) {
        warpPrefixScan(ci, co, i);
        auto laneId = threadIdx.x & (warpSize-1);
        auto warpId = i / warpSize;
        assert(warpId < warpSize);
        if ((warpSize-1) == laneId)
          ws[warpId] = co[i];
      }
      __syncthreads();
      if (size <= warpSize)
        return;
      if (threadIdx.x < warpSize)
        warpPrefixScan(ws, threadIdx.x);
      __syncthreads();
      for (auto i = first + warpSize; i < size; i += static_cast<uint32_t>(blockDim.x)) {
        auto warpId = i / warpSize;
        co[i] += ws[warpId - 1];
      }
      __syncthreads();
#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif
    }

    // same as above, may remove
    // limited to 32*32 elements....
    template <typename T>
    __host__ __device__ __forceinline__ void blockPrefixScan(T* c,
                                                             uint32_t size,
                                                             T* ws
#ifndef __HIP_DEVICE_COMPILE__
                                                             = nullptr
#endif
    ) {
#ifdef __HIP_DEVICE_COMPILE__
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDim.x % warpSize);
      uint32_t first = threadIdx.x;

      for (auto i = first; i < size; i += static_cast<uint32_t>(blockDim.x)) {
        warpPrefixScan(c, i);
        auto laneId = threadIdx.x & (warpSize-1);
        auto warpId = i / warpSize;
        assert(warpId < warpSize);
        if (warpSize-1 == laneId)
          ws[warpId] = c[i];
      }
      __syncthreads();
      if (size <= warpSize)
        return;
      if (threadIdx.x < warpSize)
        warpPrefixScan(ws, threadIdx.x);
      __syncthreads();
      for (auto i = first + warpSize; i < size; i += static_cast<uint32_t>(blockDim.x)) {
        auto warpId = i / warpSize;
        c[i] += ws[warpId - 1];
      }
      __syncthreads();
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif
    }

#ifdef TODO
#ifdef __HIP_DEVICE_COMPILE__
    // see https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
    __device__ __forceinline__ unsigned dynamic_smem_size() {
      unsigned ret;
      asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
      return ret;
    }
#endif
#endif

    // in principle not limited....
    template <typename T>
    __global__ void multiBlockPrefixScan(T const* ici, T* ico, int32_t size, int32_t* pc) {
      volatile T const* ci = ici;
      volatile T* co = ico;
      __shared__ T ws[warpSize];
#ifdef TODO
#ifdef __HIP_DEVICE_COMPILE__
      assert(sizeof(T) * gridDim.x <= dynamic_smem_size());  // size of psum below
#endif
#endif
      assert(static_cast<int32_t>(blockDim.x * gridDim.x) >= size);
      // first each block does a scan
      int off = blockDim.x * blockIdx.x;
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(blockDim.x), size - off), ws);

      // count blocks that finished
      __shared__ bool isLastBlockDone;
      if (0 == threadIdx.x) {
        __threadfence();
        auto value = atomicAdd(pc, 1);  // block counter
        isLastBlockDone = (value == (int(gridDim.x) - 1));
      }

      __syncthreads();

      if (!isLastBlockDone)
        return;

      assert(int(gridDim.x) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      HIP_DYNAMIC_SHARED( T, psum)
      for (int i = threadIdx.x, ni = gridDim.x; i < ni; i += blockDim.x) {
        int32_t j = blockDim.x * i + blockDim.x - 1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      __syncthreads();
      blockPrefixScan(psum, psum, gridDim.x, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = threadIdx.x + blockDim.x, k = 0; i < size; i += blockDim.x, ++k) {
        co[i] += psum[k];
      }
    }
  }  // namespace hip
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
