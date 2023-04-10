#ifndef HeterogeneousCoreSYCLUtilities_radixSort_H
#define HeterogeneousCoreSYCLUtilities_radixSort_H

#include <cstdint>
#include <type_traits>

#include <CL/sycl.hpp>

#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"

template <typename T>
inline void dummyReorder(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<1> item) {}

template <typename T>
inline void reorderSigned(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<1> item) {
  //move negative first...

  uint32_t first = item.get_local_id(0);
  auto firstNegbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
  uint32_t* firstNeg = (uint32_t*)firstNegbuff.get();
  *firstNeg = a[ind[0]] < 0 ? 0 : size;
  sycl::group_barrier(item.get_group());

  // find first negative
  for (auto i = first; i < size - 1; i += item.get_local_range(0)) {
    if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
      *firstNeg = i + 1;
  }

  sycl::group_barrier(item.get_group());

  auto ii = first;
  for (auto i = *firstNeg + item.get_local_id(0); i < size; i += item.get_local_range(0)) {
    ind2[ii] = ind[i];
    ii += item.get_local_range(0);
  }
  sycl::group_barrier(item.get_group());
  ii = size - *firstNeg + item.get_local_id(0);
  assert(ii >= 0);
  for (auto i = first; i < *firstNeg; i += item.get_local_range(0)) {
    ind2[ii] = ind[i];
    ii += item.get_local_range(0);
  }
  sycl::group_barrier(item.get_group());
  for (auto i = first; i < size; i += item.get_local_range(0))
    ind[i] = ind2[i];
}

template <typename T>
inline void reorderFloat(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<1> item) {
  //move negative first...

  auto firstNegbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
  uint32_t* firstNeg = (uint32_t*)firstNegbuff.get();
  *firstNeg = a[ind[0]] < 0 ? 0 : size;
  sycl::group_barrier(item.get_group());

  // find first negative
  for (uint32_t i = item.get_local_id(0); i < size - 1; i += item.get_local_range(0)) {
    if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
      *firstNeg = i + 1;
  }

  sycl::group_barrier(item.get_group());

  int ii = size - *firstNeg - item.get_local_id(0) - 1;
  for (uint32_t i = *firstNeg + item.get_local_id(0); i < size; i += item.get_local_range(0)) {
    ind2[ii] = ind[i];
    ii -= item.get_local_range(0);
  }
  sycl::group_barrier(item.get_group());
  ii = size - *firstNeg + item.get_local_id(0);
  assert(ii >= 0);
  for (uint32_t i = item.get_local_id(0); i < *firstNeg; i += item.get_local_range(0)) {
    ind2[ii] = ind[i];
    ii += item.get_local_range(0);
  }
  sycl::group_barrier(item.get_group());
  for (uint32_t i = item.get_local_id(0); i < size; i += item.get_local_range(0))
    ind[i] = ind2[i];
}

template <typename T,  // shall be interger
          int NS>      // number of significant bytes to use in sorting
__attribute__((always_inline)) void radixSortImpl(
    T const* __restrict__ a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<1> item) {
  constexpr int d = 8, w = 8 * sizeof(T);
  constexpr int sb = 1 << d;
  constexpr int ps = int(sizeof(T)) - NS;

  auto cbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[sb]>(item.get_group());
  int32_t* c = (int32_t*)cbuff.get();
  auto ctbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[sb]>(item.get_group());
  int32_t* ct = (int32_t*)ctbuff.get();
  auto cubuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[sb]>(item.get_group());
  int32_t* cu = (int32_t*)cubuff.get();

  // SYCL_BUG_ with accessors results are not updated unless printed 
  // since it's not reasonable to print stuff, local variables are used instead of shread ones
  // More studies (https://github.com/llvm/llvm-project/issues/59632) show that this could be related
  // to specific GPUs, so maybe accessors work on others..

  // auto ibsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
  // int* ibs = (int*)ibsbuff.get();
  // auto pbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
  // int* p = (int*)pbuff.get();

  assert(size > 0);
  assert(item.get_local_range(0) >= sb);

  // bool debug = false; // item.get_local_id(0)==0 && item.get_group(0)==5;

  int p = ps;

  auto j = ind;
  auto k = ind2;

  for (uint32_t i = item.get_local_id(0); i < size; i += item.get_local_range(0))
    j[i] = i;
  sycl::group_barrier(item.get_group());

  while ((sycl::group_barrier(item.get_group()), sycl::all_of_group(item.get_group(), p < w / d))) {
    if (item.get_local_id(0) < sb)
      c[item.get_local_id(0)] = 0;
    sycl::group_barrier(item.get_group());

    // fill bins
    for (uint32_t i = item.get_local_id(0); i < size; i += item.get_local_range(0)) {
      auto bin = (a[j[i]] >> d * p) & (sb - 1);
      cms::sycltools::atomic_fetch_add<int32_t, sycl::access::address_space::local_space, sycl::memory_scope::device>(
          &c[bin], static_cast<int32_t>(1));
    }
    sycl::group_barrier(item.get_group());

    // prefix scan "optimized"???...
    if (item.get_local_id(0) < sb) {
      auto x = c[item.get_local_id(0)];
      int laneId = item.get_local_id(0) & 0x1f;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        auto y = sycl::shift_group_right(item.get_sub_group(), x, offset);
        if (laneId >= offset)
          x += y;
      }
      ct[item.get_local_id(0)] = x;
    }
    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) < sb) {
      auto ss = (item.get_local_id(0) / 32) * 32 - 1;
      c[item.get_local_id(0)] = ct[item.get_local_id(0)];
      for (int i = ss; i > 0; i -= 32)
        c[item.get_local_id(0)] += ct[i];
    }
    /* 
    //prefix scan for the nulls  (for documentation)
    if (item.get_local_id(0)==0)
      for (int i = 1; i < sb; ++i) c[i] += c[i-1];
    */

    // broadcast
    int ibs = size - 1;
    while ((sycl::group_barrier(item.get_group()), sycl::all_of_group(item.get_group(), ibs > 0))) {
      int i = ibs - item.get_local_id(0);
      if (item.get_local_id(0) < sb) {
        cu[item.get_local_id(0)] = -1;
        ct[item.get_local_id(0)] = -1;
      }
      sycl::group_barrier(item.get_group());
      int32_t bin = -1;
      if (item.get_local_id(0) < sb) {
        if (i >= 0) {
          bin = (a[j[i]] >> d * p) & (sb - 1);
          ct[item.get_local_id(0)] = bin;
          cms::sycltools::atomic_fetch_max<int32_t, sycl::access::address_space::local_space, sycl::memory_scope::device>(
              &cu[bin], static_cast<int32_t>(i));
        }
      }

      sycl::group_barrier(item.get_group());
      if (item.get_local_id(0) < sb) {
        if (i >= 0 && i == cu[bin])  // ensure to keep them in order
          for (int ii = item.get_local_id(0); ii < sb; ++ii)
            if (ct[ii] == bin) {
              auto oi = ii - item.get_local_id(0);
              // assert(i>=oi);if(i>=oi)
              k[--c[bin]] = j[i - oi];
            }
      }
      sycl::group_barrier(item.get_group());
      if (bin >= 0)
        assert(c[bin] >= 0);
      ibs -= sb;
      assert(ibs < 0);
    }

    /*
    // broadcast for the nulls  (for documentation)
    if (item.get_local_id(0)==0)
    for (int i=size-first-1; i>=0; i--) { // =item.get_local_range(0)) {
      auto bin = (a[j[i]] >> d*p)&(sb-1);
      auto ik = atomicSub(&c[bin],1);
      k[ik-1] = j[i];
    }
    */

    sycl::group_barrier(item.get_group());
    assert(c[0] == 0);

    // swap j and k (local, ok)
    auto t = j;
    j = k;
    k = t;

    ++p;
    assert(p > ps);
  }

  if ((w != 8) && (0 == (NS & 1)))
    assert(j == ind);  // w/d is even so ind is correct

  if (j != ind)  // odd...
    for (uint32_t i = item.get_local_id(0); i < size; i += item.get_local_range(0))
      ind[i] = ind2[i];

  sycl::group_barrier(item.get_group());
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_unsigned<T>::value, T>::type* = nullptr>
__attribute__((always_inline)) void radixSort(
    T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<1> item) {
  radixSortImpl<T, NS>(a, ind, ind2, size, item);
  dummyReorder<T>(a, ind, ind2, size, item);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type* = nullptr>
__attribute__((always_inline)) void radixSort(
    T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<1> item) {
  radixSortImpl<T, NS>(a, ind, ind2, size, item);
  reorderSigned<T>(a, ind, ind2, size, item);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
__attribute__((always_inline)) void radixSort(
    T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<1> item) {
  using I = int;
  radixSortImpl<I, NS>((I const*)(a), ind, ind2, size, item);
  reorderFloat<I>((I const*)(a), ind, ind2, size, item);
}

template <typename T, int NS = sizeof(T)>
__attribute__((always_inline)) void radixSortMulti(T const* v,
                                                   uint16_t* index,
                                                   uint32_t const* offsets,
                                                   uint16_t* workspace,
                                                   sycl::nd_item<1> item,
                                                   uint16_t* ws) {  // ws in CUDA was: extern __shared__ uint16_t ws[]
                                                                    // pass an accessor here instead!

  auto a = v + offsets[item.get_group(0)];
  auto ind = index + offsets[item.get_group(0)];
  auto ind2 = nullptr == workspace ? ws : workspace + offsets[item.get_group(0)];
  auto size = offsets[item.get_group(0) + 1] - offsets[item.get_group(0)];
  assert(offsets[item.get_group(0) + 1] >= offsets[item.get_group(0)]);
  if (size > 0)
    radixSort<T, NS>(a, ind, ind2, size, item);
}

namespace cms {
  namespace sycltools {

    template <typename T, int NS = sizeof(T)>
    void radixSortMultiWrapper(T const* v,
                               uint16_t* index,
                               uint32_t const* offsets,
                               uint16_t* workspace,
                               sycl::nd_item<1> item,
                               uint16_t* ws) {
      radixSortMulti<T, NS>(v, index, offsets, workspace, item, ws);
    }

    template <typename T, int NS = sizeof(T)>
    void radixSortMultiWrapper2(T const* v,
                                uint16_t* index,
                                uint32_t const* offsets,
                                uint16_t* workspace,
                                sycl::nd_item<1> item,
                                uint16_t* ws) {
      radixSortMulti<T, NS>(v, index, offsets, workspace, item, ws);
    }

  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCoreSYCLUtilities_radixSort_H
