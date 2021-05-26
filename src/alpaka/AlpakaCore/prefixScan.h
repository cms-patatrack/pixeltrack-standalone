#ifndef HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h
#define HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h

#include <cstdint>
#include "CUDACore/CMSUnrollLoop.h"
#include "AlpakaCore/alpakaConfig.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

template <typename T>
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void warpPrefixScan(uint32_t laneId, T const* ci, T* co, uint32_t i, uint32_t mask) {
  // ci and co may be the same
  auto x = ci[i];
  CMS_UNROLL_LOOP
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void warpPrefixScan(uint32_t laneId, T* c, uint32_t i, uint32_t mask) {
  auto x = c[i];
  CMS_UNROLL_LOOP
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#endif

namespace cms {
  namespace alpakatools {
    // limited to 32*32 elements....
    template <typename T_Acc, typename T>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void blockPrefixScan(const T_Acc& acc,
                                                             T const* ci,
                                                             T* co,
                                                             uint32_t size,
                                                             T* ws
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
                                                             = nullptr
#endif
    ) {
#if defined ALPAKA_ACC_GPU_CUDA_ENABLED and __CUDA_ARCH__
      uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const gridBlockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDimension % 32);
      auto first = blockThreadIdx;
      auto mask = __ballot_sync(0xffffffff, first < size);
      auto laneId = blockThreadIdx & 0x1f;

      for (auto i = first; i < size; i += blockDimension) {
        warpPrefixScan(laneId, ci, co, i, mask);
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = co[i];
        mask = __ballot_sync(mask, i + blockDimension < size);
      }
      alpaka::syncBlockThreads(acc);
      if (size <= 32)
        return;
      if (blockThreadIdx < 32) {
        warpPrefixScan(laneId, ws, blockThreadIdx, 0xffffffff);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + 32; i < size; i += blockDimension) {
        uint32_t warpId = i / 32;
        co[i] += ws[warpId - 1];
      }
      alpaka::syncBlockThreads(acc);

#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif
    }

    template <typename T_Acc, typename T>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void blockPrefixScan(const T_Acc& acc,
                                                             T* __restrict__ c,
                                                             uint32_t size,
                                                             T* __restrict__ ws
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
                                                             = nullptr
#endif
    ) {
#if defined ALPAKA_ACC_GPU_CUDA_ENABLED and __CUDA_ARCH__
      uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const gridBlockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDimension % 32);
      auto first = blockThreadIdx;
      auto mask = __ballot_sync(0xffffffff, first < size);
      auto laneId = blockThreadIdx & 0x1f;

      for (auto i = first; i < size; i += blockDimension) {
        warpPrefixScan(laneId, c, i, mask);
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];
        mask = __ballot_sync(mask, i + blockDimension < size);
      }
      alpaka::syncBlockThreads(acc);
      if (size <= 32)
        return;
      if (blockThreadIdx < 32) {
        warpPrefixScan(laneId, ws, blockThreadIdx, 0xffffffff);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + 32; i < size; i += blockDimension) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
      alpaka::syncBlockThreads(acc);
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif
    }

    // limited to 1024*1024 elements....
    template <typename T>
    struct multiBlockPrefixScanFirstStep {
      template <typename T_Acc>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc, T const* ci, T* co, int32_t size) const {
        uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
        uint32_t const threadDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        uint32_t const blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

        auto& ws = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
        // first each block does a scan of size 1024; (better be enough blocks....)
#ifndef NDEBUG
        uint32_t const gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        assert(gridDimension / threadDimension <= 1024);
#endif
        int off = blockDimension * blockIdx * threadDimension;
        if (size - off > 0)
          blockPrefixScan(acc, ci + off, co + off, std::min(int(blockDimension * threadDimension), size - off), ws);
      }
    };

    // limited to 1024*1024 elements....
    template <typename T>
    struct multiBlockPrefixScanSecondStep {
      template <typename T_Acc>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc, T const* ci, T* co, int32_t size, int32_t numBlocks) const {
        uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
        uint32_t const threadDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

        uint32_t const threadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        auto* const psum(alpaka::getDynSharedMem<T>(acc));

        // first each block does a scan of size 1024; (better be enough blocks....)
        assert(static_cast<int32_t>(blockDimension * threadDimension) >= numBlocks);
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

        auto& ws = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
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
  namespace traits {

    //#############################################################################
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template <typename T, typename TAcc>
    struct BlockSharedMemDynSizeBytes<cms::alpakatools::multiBlockPrefixScanSecondStep<T>, TAcc> {
      //-----------------------------------------------------------------------------
      //! \return The size of the shared memory allocated for a block.
      template <typename TVec>
      ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
          cms::alpakatools::multiBlockPrefixScanSecondStep<T> const& myKernel,
          TVec const& blockThreadExtent,
          TVec const& threadElemExtent,
          T const* ci,
          T* co,
          int32_t size,
          int32_t numBlocks) -> T {
        alpaka::ignore_unused(myKernel);
        alpaka::ignore_unused(blockThreadExtent);
        alpaka::ignore_unused(threadElemExtent);
        alpaka::ignore_unused(ci);
        alpaka::ignore_unused(co);
        alpaka::ignore_unused(size);

        return static_cast<size_t>(numBlocks) * sizeof(T);
      }
    };

  }  // namespace traits
}  // namespace alpaka

#endif  // HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h
