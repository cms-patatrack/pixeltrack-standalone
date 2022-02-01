#ifndef AlpakaCore_threadfence_h
#define AlpakaCore_threadfence_h

#include <atomic>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfig.h"

// FIXME replace this with alpaka::mem_fence

namespace cms::alpakatools {

  // device-wide memory fence
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(TAcc const& acc) {
    static_assert(alpaka::meta::DependentFalseType<TAcc>::value,
                  "cms::alpakatools::threadfence<TAcc>(acc) has not been implemented for this Accelerator type.");
  }

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // device-wide memory fence
  // CPU serial implementation: no fence needed
  template <typename TDim, typename TIdx>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(alpaka::AccCpuSerial<TDim, TIdx> const& acc) {
    // serial implementation with a single thread, no fence needed
  }
#endif  // defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#if defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  // device-wide memory fence
  // CPU parallel implementation using TBB tasks: std::atomic_thread_fence()
  template <typename TDim, typename TIdx>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(alpaka::AccCpuTbbBlocks<TDim, TIdx> const& acc) {
    std::atomic_thread_fence(std::memory_order_acq_rel);
  }
#endif  // defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#if defined ALPAKA_ACC_GPU_CUDA_ENABLED && __CUDA_ARCH__
  // device-wide memory fence
  // GPU parallel implementation using CUDA: __threadfence()
  template <typename TDim, typename TIdx>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void threadfence(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc) {
    // device-only function
    __threadfence();
  }
#endif  // defined ALPAKA_ACC_GPU_CUDA_ENABLED && __CUDA_ARCH__

}  // namespace cms::alpakatools

#endif  // AlpakaCore_threadfence_h
