#pragma once

#include <alpaka/event/EventGenericThreads.hpp>
#include <alpaka/event/EventUniformCudaHipRt.hpp>

/* Additions that are expected to be merged back into upstream Alpaka:
 *   - alpaka-group/alpaka#1428: Add DevType<Event> trait
 */

namespace alpaka::traits {

  template <typename TDev>
  struct DevType<EventGenericThreads<TDev>> {
    using type = TDev;
  };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

  template <>
  struct DevType<EventUniformCudaHipRt> {
    using type = DevUniformCudaHipRt;
  };

#endif  // defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

}  // namespace alpaka::traits
