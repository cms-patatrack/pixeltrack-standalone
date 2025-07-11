#ifndef AlpakaCore_prefixScan_h
#define AlpakaCore_prefixScan_h

#include <algorithm>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/config.h"
#include "Framework/CMSUnrollLoop.h"

namespace cms {
  namespace alpakatools {

    // FIXME warpSize should be device-dependent
    constexpr uint32_t warpSize = 32;
    constexpr uint64_t warpMask = ~(~0ull << warpSize);

#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))

    template <typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(uint32_t laneId, T const* ci, T* co, uint32_t i, uint32_t mask) {
#if defined(__HIP_DEVICE_COMPILE__)
      ALPAKA_ASSERT_ACC(mask == warpMask);
#endif
      // ci and co may be the same
      auto x = ci[i];
      CMS_UNROLL_LOOP
      for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
#if defined(__CUDA_ARCH__)
        auto y = __shfl_up_sync(mask, x, offset);
#elif defined(__HIP_DEVICE_COMPILE__)
        auto y = __shfl_up(x, offset);
#endif
        if (laneId >= offset)
          x += y;
      }
      co[i] = x;
    }

    template <typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(uint32_t laneId, T* c, uint32_t i, uint32_t mask) {
#if defined(__HIP_DEVICE_COMPILE__)
      ALPAKA_ASSERT_ACC(mask == warpMask);
#endif
      auto x = c[i];
      CMS_UNROLL_LOOP
      for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
#if defined(__CUDA_ARCH__)
        auto y = __shfl_up_sync(mask, x, offset);
#elif defined(__HIP_DEVICE_COMPILE__)
        auto y = __shfl_up(x, offset);
#endif
        if (laneId >= offset)
          x += y;
      }
      c[i] = x;
    }

#endif  // (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))

    // limited to warpSize² elements
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void blockPrefixScan(
        const TAcc& acc, T const* ci, T* co, uint32_t size, T* ws = nullptr) {
#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
      uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      ALPAKA_ASSERT_ACC(ws);
      ALPAKA_ASSERT_ACC(size <= warpSize * warpSize);
      ALPAKA_ASSERT_ACC(0 == blockDimension % warpSize);
      auto first = blockThreadIdx;
#if defined(__CUDA_ARCH__)
      auto mask = __ballot_sync(warpMask, first < size);
#elif defined(__HIP_DEVICE_COMPILE__)
      auto mask = warpMask;
#endif
      auto laneId = blockThreadIdx & (warpSize - 1);

      for (auto i = first; i < size; i += blockDimension) {
        warpPrefixScan(laneId, ci, co, i, mask);
        auto warpId = i / warpSize;
        // FIXME test ?
        ALPAKA_ASSERT_ACC(warpId < warpSize);
        if ((warpSize - 1) == laneId)
          ws[warpId] = co[i];
#if defined(__CUDA_ARCH__)
        mask = __ballot_sync(mask, i + blockDimension < size);
#endif
      }
      alpaka::syncBlockThreads(acc);
      if (size <= warpSize)
        return;
      if (blockThreadIdx < warpSize) {
        warpPrefixScan(laneId, ws, blockThreadIdx, warpMask);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + warpSize; i < size; i += blockDimension) {
        uint32_t warpId = i / warpSize;
        co[i] += ws[warpId - 1];
      }
      alpaka::syncBlockThreads(acc);
#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif  // (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    }

    template <typename TAcc, typename T>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void blockPrefixScan(const TAcc& acc,
                                                             T* __restrict__ c,
                                                             uint32_t size,
                                                             T* __restrict__ ws = nullptr) {
#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
      uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      ALPAKA_ASSERT_ACC(ws);
      ALPAKA_ASSERT_ACC(size <= warpSize * warpSize);
      ALPAKA_ASSERT_ACC(0 == blockDimension % warpSize);
      auto first = blockThreadIdx;
#if defined(__CUDA_ARCH__)
      auto mask = __ballot_sync(warpMask, first < size);
#elif defined(__HIP_DEVICE_COMPILE__)
      auto mask = warpMask;
#endif
      auto laneId = blockThreadIdx & (warpSize - 1);

      for (auto i = first; i < size; i += blockDimension) {
        warpPrefixScan(laneId, c, i, mask);
        auto warpId = i / warpSize;
        ALPAKA_ASSERT_ACC(warpId < warpSize);
        if ((warpSize - 1) == laneId)
          ws[warpId] = c[i];
#if defined(__CUDA_ARCH__)
        mask = __ballot_sync(mask, i + blockDimension < size);
#endif
      }
      alpaka::syncBlockThreads(acc);
      if (size <= warpSize)
        return;
      if (blockThreadIdx < warpSize) {
        warpPrefixScan(laneId, ws, blockThreadIdx, warpMask);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + warpSize; i < size; i += blockDimension) {
        auto warpId = i / warpSize;
        c[i] += ws[warpId - 1];
      }
      alpaka::syncBlockThreads(acc);
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif  // (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    }

    // limited to warpSize⁴ elements
    template <typename T>
    struct multiBlockPrefixScanFirstStep {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc, T const* ci, T* co, int32_t size) const {
        uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
        uint32_t const threadDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        uint32_t const blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

        auto& ws = alpaka::declareSharedVar<T[warpSize], __COUNTER__>(acc);
        // first each block does a scan of size warpSize² (better be enough blocks)
#ifndef NDEBUG
        [[maybe_unused]] uint32_t const gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        ALPAKA_ASSERT_ACC(gridDimension / threadDimension <= (warpSize * warpSize));
#endif
#if 0
        // this is not yet available in alpaka, see
        // https://github.com/alpaka-group/alpaka/issues/1648
        ALPAKA_ASSERT_ACC(sizeof(T) * gridDimension <= dynamic_smem_size());  // size of psum below
#endif
        int off = blockDimension * blockIdx * threadDimension;
        if (size - off > 0)
          blockPrefixScan(acc, ci + off, co + off, std::min(int(blockDimension * threadDimension), size - off), ws);
      }
    };

    // limited to warpSize⁴ elements
    template <typename T>
    struct multiBlockPrefixScanSecondStep {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc, T const* ci, T* co, int32_t size, int32_t numBlocks) const {
        uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
        uint32_t const threadDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        uint32_t const threadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        T* const psum = alpaka::getDynSharedMem<T>(acc);

        // first each block does a scan of size warpSize² (better be enough blocks)
        ALPAKA_ASSERT_ACC(static_cast<int32_t>(blockDimension * threadDimension) >= numBlocks);
        for (int elemId = 0; elemId < static_cast<int>(threadDimension); ++elemId) {
          int index = +threadIdx * threadDimension + elemId;

          if (index < numBlocks) {
            int lastElementOfPreviousBlockId = index * blockDimension * threadDimension - 1;
            psum[index] = (lastElementOfPreviousBlockId < size and lastElementOfPreviousBlockId >= 0)
                              ? co[lastElementOfPreviousBlockId]
                              : T(0);
          }
        }

        alpaka::syncBlockThreads(acc);

        auto& ws = alpaka::declareSharedVar<T[warpSize], __COUNTER__>(acc);
        blockPrefixScan(acc, psum, psum, numBlocks, ws);

        for (int elemId = 0; elemId < static_cast<int>(threadDimension); ++elemId) {
          int first = threadIdx * threadDimension + elemId;
          for (int i = first + blockDimension * threadDimension; i < size; i += blockDimension * threadDimension) {
            auto k = i / (blockDimension * threadDimension);
            co[i] += psum[k];
          }
        }
      }
    };

  }  // namespace alpakatools
}  // namespace cms

namespace alpaka {
  namespace trait {

    //#############################################################################
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template <typename T, typename TAcc>
    struct BlockSharedMemDynSizeBytes<cms::alpakatools::multiBlockPrefixScanSecondStep<T>, TAcc> {
      //-----------------------------------------------------------------------------
      //! \return The size of the shared memory allocated for a block.
      template <typename TVec>
      ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
          cms::alpakatools::multiBlockPrefixScanSecondStep<T> const& /* myKernel */,
          TVec const& /* blockThreadExtent */,
          TVec const& /* threadElemExtent */,
          T const* /* ci */,
          T* /* co */,
          int32_t /* size */,
          int32_t numBlocks) -> T {
        return static_cast<size_t>(numBlocks) * sizeof(T);
      }
    };
  }  // namespace trait
}  // namespace alpaka

#endif  // AlpakaCore_prefixScan_h
