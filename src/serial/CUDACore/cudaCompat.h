#ifndef HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h
#define HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h

/*
 * Everything you need to run cuda code in plain sequential c++ code
 */

#ifndef __CUDACC__

#include <algorithm>
#include <cstdint>
#include <cstring>



using cudaStream_t = void*;
constexpr cudaStream_t cudaStreamDefault = nullptr;

namespace cms {
  namespace cudacompat {

    template <typename T1, typename T2>
    T1 atomicCAS(T1* address, T1 compare, T2 val) {
      T1 old = *address;
      *address = old == compare ? val : old;
      return old;
    }

    template <typename T1, typename T2>
    T1 atomicInc(T1* a, T2 b) {
      auto ret = *a;
      if ((*a) < T1(b))
        (*a)++;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicAdd(T1* a, T2 b) {
      auto ret = *a;
      (*a) += b;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicSub(T1* a, T2 b) {
      auto ret = *a;
      (*a) -= b;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicMin(T1* a, T2 b) {
      auto ret = *a;
      *a = std::min(*a, T1(b));
      return ret;
    }
    template <typename T1, typename T2>
    T1 atomicMax(T1* a, T2 b) {
      auto ret = *a;
      *a = std::max(*a, T1(b));
      return ret;
    }

    inline void __syncthreads() {}
    inline void __threadfence() {}
    inline bool __syncthreads_or(bool x) { return x; }
    inline bool __syncthreads_and(bool x) { return x; }
    template <typename T>
    inline T __ldg(T const* x) {
      return *x;
    }
  }  // namespace cudacompat
}  // namespace cms

// make the cudacompat implementation available in the global namespace
using namespace cms::cudacompat;

#endif  // __CUDACC__

#endif  // HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h
