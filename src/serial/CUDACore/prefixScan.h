#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cstdint>

#include "CUDACore/cudaCompat.h"
#include "CUDACore/cuda_assert.h"

namespace cms {
  namespace cuda {

    // limited to 32*32 elements....
    template <typename VT>
    void blockPrefixScan(VT const* ci, VT* co, uint32_t size) {
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
    }

    // same as above, may remove
    // limited to 32*32 elements....
    template <typename T>
    void blockPrefixScan(T* c, uint32_t size) {
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
    }

    // in principle not limited....
    template <typename T>
    void multiBlockPrefixScan(T const* ici, T* ico, int32_t size, int32_t* pc) {
      volatile T const* ci = ici;
      volatile T* co = ico;
      T ws[32];
      assert(1 >= size);
      // first each block does a scan
      int off = 0;
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(1), size - off), ws);

      // count blocks that finished
      bool isLastBlockDone;

      auto value = atomicAdd(pc, 1);  // block counter
      isLastBlockDone = (value == 0);

      if (!isLastBlockDone)
        return;

      assert(int(1) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      extern T psum[];
      for (int i = 0, ni = 1; i < ni; i++) {
        auto j = i;
        psum[i] = (j < size) ? co[j] : T(0);
      }

      blockPrefixScan(psum, psum, 1, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = 1, k = 0; i < size; i++, ++k) {
        co[i] += psum[k];
      }
    }
  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
