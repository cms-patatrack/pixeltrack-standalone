#ifndef HeterogeneousCore_SYCLUtilities_interface_prefixScan_h
#define HeterogeneousCore_SYCLUtilities_interface_prefixScan_h

#include <cstdint>
#include <sycl/sycl.hpp>
#include "SYCLCore/syclAtomic.h"
#include "SYCLCore/sycl_assert.h"

template <typename T>
void __attribute__((always_inline))
warpPrefixScan(T const* __restrict__ ci, T* __restrict__ co, uint32_t i, sycl::nd_item<1> item) {
  // ci and co may be the same
  auto x = ci[i];
  int laneId = item.get_local_id(0) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = sycl::shift_group_right(item.get_sub_group(), x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
void __attribute__((always_inline)) warpPrefixScan(T* c, uint32_t i, sycl::nd_item<1> item) {
  auto x = c[i];
  int laneId = item.get_local_id(0) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = sycl::shift_group_right(item.get_sub_group(), x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

namespace cms {
  namespace sycltools {

    // 1) limited to 32*32 elements....
    template <typename VT, typename T>
    __attribute__((always_inline)) void blockPrefixScan(
        VT const* ci, VT* co, uint32_t size, sycl::nd_item<1> item, T* ws) {
      auto first = item.get_local_id(0);

      //__ballot_sync in CUDA
      // size_t id = item.get_sub_group().get_local_linear_id();
      // uint32_t local_val = (first < size ? 1u : 0u) << id;
      // auto mask = sycl::reduce_over_group(item.get_sub_group(), local_val, sycl::plus<>());
      //end of __ballot_sync equivalent

      for (auto i = first; i < size; i += item.get_local_range(0)) {
        warpPrefixScan(ci, co, i, item);
        int laneId = item.get_local_id(0) & 0x1f;
        auto warpId = i / 32;
        if (31 == laneId)
          ws[warpId] = co[i];

        //__ballot_sync in CUDA
        // size_t id2 = item.get_sub_group().get_local_linear_id();
        // uint32_t local_val2 = ((i + item.get_local_range(0)) < size ? 1u : 0u) << id2;
        // mask = sycl::reduce_over_group(item.get_sub_group(), local_val2, sycl::plus<>());
        //end of __ballot_sync equivalent
      }
      // item.barrier(sycl::access::fence_space::local_space);
      sycl::group_barrier(item.get_group());

      if (size <= 32)
        return;

      if (item.get_local_id(0) < 32)
        warpPrefixScan(ws, item.get_local_id(0), item);

      // item.barrier(sycl::access::fence_space::local_space);
      sycl::group_barrier(item.get_group());

      for (auto i = first + 32; i < size; i += item.get_local_range(0)) {
        auto warpId = i / 32;
        co[i] += ws[warpId - 1];
      }

      // item.barrier(sycl::access::fence_space::local_space);
      sycl::group_barrier(item.get_group());
    }

    // same as above (1), may remove
    // limited to 32*32 elements....
    template <typename T>
    __attribute__((always_inline)) void blockPrefixScan(T* c, uint32_t size, sycl::nd_item<1> item, T* ws) {
      auto first = item.get_local_id(0);

      //__ballot_sync in CUDA
      // size_t id = item.get_sub_group().get_local_linear_id();
      // uint32_t local_val = (first < size ? 1u : 0u) << id;
      // auto mask = sycl::reduce_over_group(item.get_sub_group(), local_val, sycl::plus<>());
      //end of __ballot_sync equivalent

      for (auto i = first; i < size; i += item.get_local_range(0)) {
        warpPrefixScan(c, i, item);
        int laneId = item.get_local_id(0) & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];

        //__ballot_sync in CUDA
        // size_t id2 = item.get_sub_group().get_local_linear_id();
        // uint32_t local_val2 = ((i + item.get_local_range(0)) < size ? 1u : 0u) << id2;
        // mask = sycl::reduce_over_group(item.get_sub_group(), local_val2, sycl::plus<>());
        //end of __ballot_sync equivalent
      }
      // item.barrier(sycl::access::fence_space::local_space);
      sycl::group_barrier(item.get_group());

      if (size <= 32)
        return;

      if (item.get_local_id(0) < 32)
        warpPrefixScan(ws, item.get_local_id(0), item);

      // item.barrier(sycl::access::fence_space::local_space);
      sycl::group_barrier(item.get_group());
      for (auto i = first + 32; i < size; i += item.get_local_range(0)) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
      // item.barrier(sycl::access::fence_space::local_space);
      sycl::group_barrier(item.get_group());
    }

    // // see https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
    // __attribute__((always_inline)) unsigned dynamic_smem_size() {
    //   unsigned ret;
    //   asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret)); // TODO_
    //   return ret;
    // }

    // in principle not limited....
    template <typename T>
    void multiBlockPrefixScan(T const* ici, T* ico, int32_t size, int32_t* pc, sycl::nd_item<1> item, T* psum) {
      volatile T const* ci = ici;
      volatile T* co = ico;
      auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T[32]>(item.get_group());
      T* ws = (T*)wsbuff.get();

      // assert(sizeof(T) * item.get_group_range(0) <= dynamic_smem_size());  // size of psum below TODO_
      assert((int32_t)(item.get_local_range(0) * item.get_group_range(0)) >= size);
      // first each block does a scan
      int off = item.get_local_range(0) * item.get_group(0);
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(item.get_local_range(0)), size - off), item, ws);

      // count blocks that finished
      auto isLastBlockDonebuff = sycl::ext::oneapi::group_local_memory_for_overwrite<bool>(item.get_group());
      bool* isLastBlockDone = (bool*)isLastBlockDonebuff.get();

      if (0 == item.get_local_id(0)) {
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
        auto value = cms::sycltools::atomic_fetch_add<int32_t,
                                                      sycl::access::address_space::global_space,
                                                      sycl::memory_scope::device>(pc, static_cast<int32_t>(1));
        *isLastBlockDone = (value == (int(item.get_group_range(0)) - 1));
      }

      // item.barrier(sycl::access::fence_space::local_space);
      sycl::group_barrier(item.get_group());

      if (!(*isLastBlockDone))
        return;

      assert(int(item.get_group_range(0)) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      // auto psumbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T[nblocks]>(item.get_group());
      // T* psum = (T*)psumbuff.get();

      for (int i = item.get_local_id(0), ni = item.get_group_range(0); i < ni; i += item.get_local_range(0)) {
        int32_t j = item.get_local_range(0) * i + item.get_local_range(0) - 1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      //Same as above (0)
      // item.barrier(sycl::access::fence_space::local_space);
      sycl::group_barrier(item.get_group());
      blockPrefixScan(psum, psum, item.get_group_range(0), item, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = item.get_local_id(0) + item.get_local_range(0), k = 0; i < size; i += item.get_local_range(0), ++k) {
        co[i] += psum[k];
      }
    }
  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLUtilities_interface_prefixScan_h
