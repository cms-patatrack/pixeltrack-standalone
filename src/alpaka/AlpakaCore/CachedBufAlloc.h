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
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevCpu, alpaka::QueueCudaRtNonBlocking, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevCpu const& dev,
                                                alpaka::QueueCudaRtNonBlocking queue,
                                                TExtent const& extent) -> alpaka::BufCpu<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getHostCachingAllocator<alpaka::QueueCudaRtNonBlocking>();

        // FIXME the BufCpu does not support a pitch ?
        size_t size = alpaka::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufCpu<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), extent);
      }
    };

    //! The caching memory allocator implementation for the CUDA device
    template <typename TElem, typename TDim, typename TIdx, typename TQueue>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevCudaRt, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevCudaRt const& dev, TQueue queue, TExtent const& extent)
          -> alpaka::BufCudaRt<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getDeviceCachingAllocator<alpaka::DevCudaRt, TQueue>(dev);

        size_t width = alpaka::getWidth(extent);
        size_t widthBytes = width * static_cast<TIdx>(sizeof(TElem));
        // TODO implement pitch for TDim > 1
        size_t pitchBytes = widthBytes;
        size_t size = alpaka::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufCudaRt<TElem, TDim, TIdx>(
            dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), pitchBytes, extent);
      }
    };

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

    //! The caching memory allocator implementation for the pinned host memory
    template <typename TElem, typename TDim, typename TIdx>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevCpu, alpaka::QueueHipRtNonBlocking, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevCpu const& dev,
                                                alpaka::QueueHipRtNonBlocking queue,
                                                TExtent const& extent) -> alpaka::BufCpu<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getHostCachingAllocator<alpaka::QueueHipRtNonBlocking>();

        // FIXME the BufCpu does not support a pitch ?
        size_t size = alpaka::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufCpu<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), extent);
      }
    };

    //! The caching memory allocator implementation for the ROCm/HIP device
    template <typename TElem, typename TDim, typename TIdx, typename TQueue>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevHipRt, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevHipRt const& dev, TQueue queue, TExtent const& extent)
          -> alpaka::BufHipRt<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getDeviceCachingAllocator<alpaka::DevHipRt, TQueue>(dev);

        size_t width = alpaka::getWidth(extent);
        size_t widthBytes = width * static_cast<TIdx>(sizeof(TElem));
        // TODO implement pitch for TDim > 1
        size_t pitchBytes = widthBytes;
        size_t size = alpaka::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufHipRt<TElem, TDim, TIdx>(
            dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), pitchBytes, extent);
      }
    };

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_SYCL_ONEAPI_CPU

    //! The caching memory allocator implementation for the pinned host memory
    template <typename TElem, typename TDim, typename TIdx>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevCpu, alpaka::QueueCpuSyclNonBlocking, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevCpu const& dev,
                                                alpaka::QueueCpuSyclNonBlocking queue,
                                                TExtent const& extent) -> alpaka::BufCpu<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getHostCachingAllocator<alpaka::QueueCpuSyclNonBlocking>();

        // FIXME the BufCpu does not support a pitch ?
        size_t size = alpaka::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufCpu<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), extent);
      }
    };

    //! The caching memory allocator implementation for the SYCL CPU device
    template <typename TElem, typename TDim, typename TIdx, typename TQueue>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevCpuSycl, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevCpuSycl const& dev, TQueue queue, TExtent const& extent)
          -> alpaka::BufCpuSycl<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getDeviceCachingAllocator<alpaka::DevCpuSycl, TQueue>(dev);

        // size_t width = alpaka::getWidth(extent);
        // size_t widthBytes = width * static_cast<TIdx>(sizeof(TElem));
        // TODO implement pitch in SYCL
        // size_t pitchBytes = widthBytes;
        size_t size = alpaka::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufCpuSycl<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), extent);
      }
    };

#endif  // ALPAKA_SYCL_ONEAPI_CPU

#ifdef ALPAKA_SYCL_ONEAPI_GPU

    //! The caching memory allocator implementation for the pinned host memory
    template <typename TElem, typename TDim, typename TIdx>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevCpu, alpaka::QueueGpuSyclIntelNonBlocking, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevCpu const& dev,
                                                alpaka::QueueGpuSyclIntelNonBlocking queue,
                                                TExtent const& extent) -> alpaka::BufCpu<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getHostCachingAllocator<alpaka::QueueGpuSyclIntelNonBlocking>();

        // FIXME the BufCpu does not support a pitch ?
        size_t size = alpaka::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufCpu<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), extent);
      }
    };

    //! The caching memory allocator implementation for the SYCL GPU device
    template <typename TElem, typename TDim, typename TIdx, typename TQueue>
    struct CachedBufAlloc<TElem, TDim, TIdx, alpaka::DevGpuSyclIntel, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static auto allocCachedBuf(alpaka::DevGpuSyclIntel const& dev, TQueue queue, TExtent const& extent)
          -> alpaka::BufGpuSyclIntel<TElem, TDim, TIdx> {
        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

        auto& allocator = getDeviceCachingAllocator<alpaka::DevGpuSyclIntel, TQueue>(dev);

        // size_t width = alpaka::getWidth(extent);
        // size_t widthBytes = width * static_cast<TIdx>(sizeof(TElem));
        // TODO implement pitch in SYCL
        // size_t pitchBytes = widthBytes;
        size_t size = alpaka::getExtentProduct(extent);
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };

        return alpaka::BufGpuSyclIntel<TElem, TDim, TIdx>(
            dev, reinterpret_cast<TElem*>(memPtr), std::move(deleter), extent);
      }
    };

#endif  // ALPAKA_SYCL_ONEAPI_GPU

  }  // namespace traits

  template <typename TElem, typename TIdx, typename TExtent, typename TQueue, typename TDev>
  ALPAKA_FN_HOST auto allocCachedBuf(TDev const& dev, TQueue queue, TExtent const& extent = TExtent()) {
    return traits::CachedBufAlloc<TElem, alpaka::Dim<TExtent>, TIdx, TDev, TQueue>::allocCachedBuf(dev, queue, extent);
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_CachedBufAlloc_h
