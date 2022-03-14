#pragma once

#include <alpaka/alpaka.hpp>

// TODO move this into Alpaka
namespace alpaka {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  using PltfCudaRt = PltfUniformCudaHipRt;

  using EventCudaRt = EventUniformCudaHipRt;

  template <typename TElem, typename TDim, typename TIdx>
  using BufCudaRt = BufUniformCudaHipRt<TElem, TDim, TIdx>;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  using PltfHipRt = PltfUniformCudaHipRt;

  using EventHipRt = EventUniformCudaHipRt;

  template <typename TElem, typename TDim, typename TIdx>
  using BufHipRt = BufUniformCudaHipRt<TElem, TDim, TIdx>;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

}  // namespace alpaka
