#ifndef AlpakaCore_CachedBufAlloc_h
#define AlpakaCore_CachedBufAlloc_h

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/getDeviceCachingAllocator.h"
#include "AlpakaCore/getHostCachingAllocator.h"
#include "Framework/demangle.h"

namespace cms::alpakatools {

  namespace traits {

    //! The caching memory allocator trait.
    template <typename TElem, typename TDim, typename TIdx, typename TDev, typename TQueue, typename TSfinae = void>
    struct CachedBufAlloc {
      static_assert(alpaka::meta::DependentFalseType<TDev>::value, "This device does not support a caching allocator");
    };

    //! The caching memory allocator implementation for the CPU device
    template <typename TElem, typename TDim, typename TIdx, typename TQueue>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevCpu, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevCpu const& dev, TQueue queue, TExtent const& extent)
          -> alpaka::BufCpu<TElem, TDim, TIdx> {
        // non-cached host-only memory
        return alpaka::allocAsyncBuf<TElem, TIdx>(queue, extent);
      }
    };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

    //! The caching memory allocator implementation for the pinned host memory
    template <typename TElem, typename TDim, typename TIdx>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevCpu, alpaka::QueueUniformCudaHipRtNonBlocking, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevCpu const& dev,
                                                alpaka::QueueUniformCudaHipRtNonBlocking queue,
                                                TExtent const& extent) -> alpaka::BufCpu<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getHostCachingAllocator<alpaka::QueueUniformCudaHipRtNonBlocking>();

        // FIXME the BufCpu does not support a pitch ?
        size_t size = alpaka::extent::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufCpu<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), extent);
      }
    };

    //! The caching memory allocator implementation for the CUDA/HIP device
    template <typename TElem, typename TDim, typename TIdx, typename TQueue>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevUniformCudaHipRt, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevUniformCudaHipRt const& dev,
                                                TQueue queue,
                                                TExtent const& extent)
          -> alpaka::BufUniformCudaHipRt<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getDeviceCachingAllocator<alpaka::DevUniformCudaHipRt, TQueue>(dev);

        size_t width = alpaka::extent::getWidth(extent);
        size_t widthBytes = width * static_cast<TIdx>(sizeof(TElem));
        // TODO implement pitch for TDim > 1
        size_t pitchBytes = widthBytes;
        size_t size = alpaka::extent::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufUniformCudaHipRt<TElem, TDim, TIdx>(
            dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), pitchBytes, extent);
      }
    };

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

  }  // namespace traits

  template <typename TElem, typename TIdx, typename TExtent, typename TQueue, typename TDev>
  ALPAKA_FN_HOST auto allocCachedBuf(TDev const& dev, TQueue queue, TExtent const& extent = TExtent()) {
    return traits::CachedBufAlloc<TElem, alpaka::Dim<TExtent>, TIdx, TDev, TQueue>::allocCachedBuf(dev, queue, extent);
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_CachedBufAlloc_h
