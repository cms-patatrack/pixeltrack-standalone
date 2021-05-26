#ifndef HeterogeneousCoreCUDAUtilities_threadfence_h
#define HeterogeneousCoreCUDAUtilities_threadfence_h

#include <type_traits>

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/alpakaKernelCommon.h"

namespace cms {
  namespace alpakatools {

    // device-wide memory fence
    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(T_Acc const& acc) {
      static_assert(std::is_same_v<T_Acc, void>, "cms::alpakatools::threadfence<T_Acc>(acc) has not been implemented for this Accelerator type.");
    }

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    // device-wide memory fence
    // CPU serial implementation: no fence needed
    template <typename TDim, typename TIdx>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(alpaka::AccCpuSerial<TDim, TIdx> const& acc) {
      // serial implementation with a single thread, no fence needed
    }
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    // device-wide memory fence
    // CPU parallel implementation using TBB tasks: std::atomic_thread_fence()
    template <typename TDim, typename TIdx>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(alpaka::AccCpuTbbBlocks<TDim, TIdx> const& acc) {
      std::atomic_thread_fence(std::memory_order_acq_rel);
    }
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // device-wide memory fence
    // GPU parallel implementation using CUDA: __threadfence()
    template <typename TDim, typename TIdx>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc) {
      // device-only function
      __threadfence();
    }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

  }  // namespace alpakatools
}  // namespace cms

#endif  // HeterogeneousCoreCUDAUtilities_threadfence_h
