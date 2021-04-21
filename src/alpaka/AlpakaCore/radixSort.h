#ifndef HeterogeneousCoreCUDAUtilities_radixSort_H
#define HeterogeneousCoreCUDAUtilities_radixSort_H

#include <cstdint>
#include <type_traits>

#include "AlpakaCore/alpakaKernelCommon.h"

namespace cms {
  namespace alpakatools {

    template <typename T_Acc, typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void dummyReorder(
        const T_Acc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {}

    template <typename T_Acc, typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void reorderSigned(
        const T_Acc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
      //move negative first...

      auto& firstNeg = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      firstNeg = a[ind[0]] < 0 ? 0 : size;
      alpaka::syncBlockThreads(acc);

      // find first negative
      cms::alpakatools::for_each_element_in_block_strided(acc, size - 1, [&](uint32_t i) {
        if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
          firstNeg = i + 1;
      });

      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block_strided(
          acc, size, firstNeg, [&](uint32_t i) { ind2[i - firstNeg] = ind[i]; });
      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block_strided(
          acc, firstNeg, [&](uint32_t i) { ind2[i + size - firstNeg] = ind[i]; });
      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block_strided(acc, size, [&](uint32_t i) { ind[i] = ind2[i]; });
    }

    template <typename T_Acc, typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void reorderFloat(
        const T_Acc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
      //move negative first...

      auto& firstNeg = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      firstNeg = a[ind[0]] < 0 ? 0 : size;
      alpaka::syncBlockThreads(acc);

      // find first negative
      cms::alpakatools::for_each_element_in_block_strided(acc, size - 1, [&](uint32_t i) {
        if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
          firstNeg = i + 1;
      });
      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block_strided(
          acc, size, firstNeg, [&](uint32_t i) { ind2[size - firstNeg - i - 1] = ind[i]; });
      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block_strided(
          acc, firstNeg, [&](uint32_t i) { ind2[i + size - firstNeg] = ind[i]; });
      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block_strided(acc, size, [&](uint32_t i) { ind[i] = ind2[i]; });
    }

    template <typename T_Acc,
              typename T,  // shall be interger
              int NS,      // number of significant bytes to use in sorting
              typename RF>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSortImpl(
        const T_Acc& acc, T const* __restrict__ a, uint16_t* ind, uint16_t* ind2, uint32_t size, RF reorder) {
      constexpr int d = 8, w = 8 * sizeof(T);
      constexpr int sb = 1 << d;
      constexpr int ps = int(sizeof(T)) - NS;

      auto& c = alpaka::declareSharedVar<int32_t[sb], __COUNTER__>(acc);
      auto& ct = alpaka::declareSharedVar<int32_t[sb], __COUNTER__>(acc);
      auto& cu = alpaka::declareSharedVar<int32_t[sb], __COUNTER__>(acc);

      auto& ibs = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& p = alpaka::declareSharedVar<int, __COUNTER__>(acc);

      assert(size > 0);

      const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      assert(blockDimension >= sb);

      // bool debug = false; // threadIdx.x==0 && blockIdx.x==5;

      p = ps;

      auto j = ind;
      auto k = ind2;

      cms::alpakatools::for_each_element_in_block_strided(acc, size, [&](uint32_t i) { j[i] = i; });
      alpaka::syncBlockThreads(acc);

      while (alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, (p < w / d))) {
        cms::alpakatools::for_each_element_in_block(acc, sb, [&](uint32_t i) { c[i] = 0; });
        alpaka::syncBlockThreads(acc);

        // fill bins
        cms::alpakatools::for_each_element_in_block_strided(acc, size, [&](uint32_t i) {
          auto bin = (a[j[i]] >> d * p) & (sb - 1);
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c[bin], 1);
        });
        alpaka::syncBlockThreads(acc);

        // prefix scan "optimized"???...
        cms::alpakatools::for_each_element_in_block(acc, sb, [&](uint32_t i) {
          auto x = c[i];
          auto laneId = i & 0x1f;

          for (int offset = 1; offset < 32; offset <<= 1) {
            auto y = __shfl_up_sync(0xffffffff, x, offset);
            if (laneId >= (uint32_t)offset)
              x += y;
          }
          ct[i] = x;
        });
        alpaka::syncBlockThreads(acc);

        cms::alpakatools::for_each_element_in_block(acc, sb, [&](uint32_t i) {
          auto ss = (i / 32) * 32 - 1;
          c[i] = ct[i];
          for (int i = ss; i > 0; i -= 32)
            c[i] += ct[i];
        });

        /* 
    //prefix scan for the nulls  (for documentation)
    if (threadIdxLocal==0)
      for (int i = 1; i < sb; ++i) c[i] += c[i-1];
    */

        // broadcast
        ibs = size - 1;
        alpaka::syncBlockThreads(acc);

        while (alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, ibs > 0)) {
          cms::alpakatools::for_each_element_in_block(acc, sb, [&](uint32_t i) {
            cu[i] = -1;
            ct[i] = -1;
          });
          alpaka::syncBlockThreads(acc);

          cms::alpakatools::for_each_element_in_block(acc, sb, [&](uint32_t idx) {
            int i = ibs - idx;
            int32_t bin = -1;
            if (i >= 0) {
              bin = (a[j[i]] >> d * p) & (sb - 1);
              ct[idx] = bin;
              alpaka::atomicOp<alpaka::AtomicMax>(acc, &cu[bin], int(i));
            }
          });
          alpaka::syncBlockThreads(acc);

          cms::alpakatools::for_each_element_in_block(acc, sb, [&](uint32_t idx) {
            int i = ibs - idx;
            int32_t bin = (i >= 0 ? ((a[j[i]] >> d * p) & (sb - 1)) : -1);
            if (i >= 0 && i == cu[bin])  // ensure to keep them in order
              for (int ii = idx; ii < sb; ++ii)
                if (ct[ii] == bin) {
                  auto oi = ii - idx;
                  // assert(i>=oi);if(i>=oi)
                  k[--c[bin]] = j[i - oi];
                }
          });
          alpaka::syncBlockThreads(acc);

          const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
          if (threadIdxLocal == 0)
            ibs -= sb;
          alpaka::syncBlockThreads(acc);
        }

        /*
    // broadcast for the nulls  (for documentation)
    if (threadIdxLocal==0)
    for (int i=size-first-1; i>=0; i--) { // =blockDim.x) {
      auto bin = (a[j[i]] >> d*p)&(sb-1);
      auto ik = atomicSub(&c[bin],1);
      k[ik-1] = j[i];
    }
    */

        alpaka::syncBlockThreads(acc);
        assert(c[0] == 0);

        // swap (local, ok)
        auto t = j;
        j = k;
        k = t;

        const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
        if (threadIdxLocal == 0)
          ++p;
        alpaka::syncBlockThreads(acc);
      }

      if ((w != 8) && (0 == (NS & 1)))
        assert(j == ind);  // w/d is even so ind is correct

      if (j != ind)  // odd...
        cms::alpakatools::for_each_element_in_block_strided(acc, size, [&](uint32_t i) { ind[i] = ind2[i]; });

      alpaka::syncBlockThreads(acc);

      // now move negative first... (if signed)
      reorder(acc, a, ind, ind2, size);
    }

    template <typename T_Acc,
              typename T,
              int NS = sizeof(T),  // number of significant bytes to use in sorting
              typename std::enable_if<std::is_unsigned<T>::value, T>::type* = nullptr>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSort(
        const T_Acc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
      radixSortImpl<T_Acc, T, NS>(acc, a, ind, ind2, size, dummyReorder<T_Acc, T>);
    }

    template <typename T_Acc,
              typename T,
              int NS = sizeof(T),  // number of significant bytes to use in sorting
              typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type* = nullptr>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSort(
        const T_Acc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
      radixSortImpl<T_Acc, T, NS>(acc, a, ind, ind2, size, reorderSigned<T_Acc, T>);
    }

    template <typename T_Acc,
              typename T,
              int NS = sizeof(T),  // number of significant bytes to use in sorting
              typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSort(
        const T_Acc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
      using I = int;
      radixSortImpl<T_Acc, I, NS>(acc, (I const*)(a), ind, ind2, size, reorderFloat<T_Acc, I>);
    }

    /* Not needed
template <typename T, int NS = sizeof(T)>
ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSortMulti(T const* v,
                                               uint16_t* index,
                                               uint32_t const* offsets,
                                               uint16_t* workspace) {

  extern __shared__ uint16_t ws[];

  auto a = v + offsets[blockIdx.x];
  auto ind = index + offsets[blockIdx.x];
  auto ind2 = nullptr == workspace ? ws : workspace + offsets[blockIdx.x];
  auto size = offsets[blockIdx.x + 1] - offsets[blockIdx.x];
  assert(offsets[blockIdx.x + 1] >= offsets[blockIdx.x]);
  if (size > 0)
    radixSort<T, NS>(a, ind, ind2, size);
}

    template <typename T, int NS = sizeof(T)>
    __global__ void __launch_bounds__(256, 4)
        radixSortMultiWrapper(T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace) {
      radixSortMulti<T, NS>(v, index, offsets, workspace);
    }

    template <typename T, int NS = sizeof(T)>
    __global__ void radixSortMultiWrapper2(T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace) {
      radixSortMulti<T, NS>(v, index, offsets, workspace);
    }
*/

  }  // namespace alpakatools
}  // namespace cms

#endif  // HeterogeneousCoreCUDAUtilities_radixSort_H
