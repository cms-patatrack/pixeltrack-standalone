#ifndef HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h
#define HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h

#include <cstdint>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/threadfence.h"
#include "Framework/CMSUnrollLoop.h"

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
      const int32_t blockDim(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const int32_t gridBlockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      const int32_t blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      ALPAKA_ASSERT_OFFLOAD(ws);
      ALPAKA_ASSERT_OFFLOAD(size <= 1024);
      ALPAKA_ASSERT_OFFLOAD(0 == blockDim % 32);
      auto first = blockThreadIdx;
      auto mask = __ballot_sync(0xffffffff, first < size);
      auto laneId = blockThreadIdx & 0x1f;

      for (auto i = first; i < size; i += blockDim) {
        warpPrefixScan(laneId, ci, co, i, mask);
        auto warpId = i / 32;
        ALPAKA_ASSERT_OFFLOAD(warpId < 32);
        if (31 == laneId)
          ws[warpId] = co[i];
        mask = __ballot_sync(mask, i + blockDim < size);
      }
      alpaka::syncBlockThreads(acc);
      if (size <= 32)
        return;
      if (blockThreadIdx < 32) {
        warpPrefixScan(laneId, ws, blockThreadIdx, 0xffffffff);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + 32; i < size; i += blockDim) {
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
      const int32_t blockDim(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const int32_t gridBlockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      const int32_t blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      ALPAKA_ASSERT_OFFLOAD(ws);
      ALPAKA_ASSERT_OFFLOAD(size <= 1024);
      ALPAKA_ASSERT_OFFLOAD(0 == blockDim % 32);
      auto first = blockThreadIdx;
      auto mask = __ballot_sync(0xffffffff, first < size);
      auto laneId = blockThreadIdx & 0x1f;

      for (auto i = first; i < size; i += blockDim) {
        warpPrefixScan(laneId, c, i, mask);
        auto warpId = i / 32;
        ALPAKA_ASSERT_OFFLOAD(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];
        mask = __ballot_sync(mask, i + blockDim < size);
      }
      alpaka::syncBlockThreads(acc);
      if (size <= 32)
        return;
      if (blockThreadIdx < 32) {
        warpPrefixScan(laneId, ws, blockThreadIdx, 0xffffffff);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + 32; i < size; i += blockDim) {
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
    struct multiBlockPrefixScan {
      template <typename T_Acc>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc, T const* ci, T* co, int32_t size, int32_t* pc) const {
        const int32_t blockDim(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
        const int32_t threadDim(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        const int32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        const int32_t threadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // first each block does a scan of size 1024; (better be enough blocks....)
        int32_t const gridDim(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        ALPAKA_ASSERT_OFFLOAD(gridDim / threadDim <= 1024);
        int off = blockDim * blockIdx * threadDim;
        auto& ws = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
        if (size - off > 0)
          blockPrefixScan(acc, ci + off, co + off, std::min(int(blockDim * threadDim), size - off), ws);

        auto& isLastBlockDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
        if (0 == threadIdx) {
          cms::alpakatools::threadfence(acc);
          auto value = alpaka::atomicAdd(acc, pc, 1, alpaka::hierarchy::Blocks{});  // block counter
          isLastBlockDone = (value == (gridDim - 1));
        }

        alpaka::syncBlockThreads(acc);

        if (!isLastBlockDone)
          return;

        ALPAKA_ASSERT_OFFLOAD(gridDim == *pc);

        auto& psum = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

        ALPAKA_ASSERT_OFFLOAD(static_cast<int32_t>(blockDim * threadDim) >= gridDim);

        for (int elemId = 0; elemId < static_cast<int>(threadDim); ++elemId) {
          int index = +threadIdx * threadDim + elemId;

          if (index < gridDim) {
            int lastElementOfPreviousBlockId = index * blockDim * threadDim - 1;
            psum[index] = (lastElementOfPreviousBlockId < size and lastElementOfPreviousBlockId >= 0)
                              ? co[lastElementOfPreviousBlockId]
                              : T(0);
          }
        }

        alpaka::syncBlockThreads(acc);
        blockPrefixScan(acc, psum, psum, gridDim, ws);

        for (int elemId = 0; elemId < static_cast<int>(threadDim); ++elemId) {
          int first = threadIdx * threadDim + elemId;
          for (int i = first + blockDim * threadDim; i < size; i += blockDim * threadDim) {
            auto k = i / (blockDim * threadDim);
            co[i] += psum[k];
          }
        }
      }
    };
  }  // namespace alpakatools
}  // namespace cms

#endif  // HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h
