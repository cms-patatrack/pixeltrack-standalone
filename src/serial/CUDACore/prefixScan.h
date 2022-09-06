#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cstdint>

#include "CUDACore/cudaCompat.h"
#include "CUDACore/cuda_assert.h"

namespace cms {
  namespace cuda {

    // limited to 32*32 elements....
    template <typename VT, typename T>
    __host__ __device__ __forceinline__ void blockPrefixScan(VT const* ci,
                                                             VT* co,
                                                             uint32_t size,
                                                             T* ws
                                                             = nullptr
    ) {
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
    }

    // same as above, may remove
    // limited to 32*32 elements....
    template <typename T>
    __host__ __device__ __forceinline__ void blockPrefixScan(T* c,
                                                             uint32_t size,
                                                             T* ws
                                                             = nullptr
    ) {
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
    }

    // in principle not limited....
    template <typename T>
    __global__ void multiBlockPrefixScan(T const* ici, T* ico, int32_t size, int32_t* pc) {
      volatile T const* ci = ici;
      volatile T* co = ico;
      __shared__ T ws[32];
      assert(blockDim.x * gridDim.x >= size);
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
      extern __shared__ T psum[];
      for (int i = threadIdx.x, ni = gridDim.x; i < ni; i += blockDim.x) {
        auto j = blockDim.x * i + blockDim.x - 1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      __syncthreads();
      blockPrefixScan(psum, psum, gridDim.x, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = threadIdx.x + blockDim.x, k = 0; i < size; i += blockDim.x, ++k) {
        co[i] += psum[k];
      }
    }
  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
