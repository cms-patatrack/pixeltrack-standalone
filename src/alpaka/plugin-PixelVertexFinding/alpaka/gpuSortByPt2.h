#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h

#include "AlpakaCore/alpakaKernelCommon.h"
#include "AlpakaCore/radixSort.h"

#include "AlpakaCore/HistoContainer.h"

#include "gpuVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace gpuVertexFinder {

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void sortByPt2(const T_Acc& acc,
                                                                                 ZVertices* pdata,
                                                                                 WorkSpace* pws) {
      auto& __restrict__ data = *pdata;
      auto& __restrict__ ws = *pws;
      auto nt = ws.ntrks;
      float const* __restrict__ ptt2 = ws.ptt2;
      uint32_t const& nvFinal = data.nvFinal;

      int32_t const* __restrict__ iv = ws.iv;
      float* __restrict__ ptv2 = data.ptv2;
      uint16_t* __restrict__ sortInd = data.sortInd;

      // if (threadIdxLocal == 0)
      //    printf("sorting %d vertices\n",nvFinal);

      if (nvFinal < 1)
        return;

      // fill indexing
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) { data.idv[ws.itrk[i]] = iv[i]; });

      // can be done asynchronoisly at the end of previous event
      cms::alpakatools::for_each_element_in_block_strided(acc, nvFinal, [&](uint32_t i) { ptv2[i] = 0; });
      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        if (iv[i] <= 9990) {
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, &ptv2[iv[i]], ptt2[i]);
        }
      });
      alpaka::syncBlockThreads(acc);

      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      if (1 == nvFinal) {
        if (threadIdxLocal == 0)
          sortInd[0] = 0;
        return;
      }
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      auto& sws = alpaka::declareSharedVar<uint16_t[1024], __COUNTER__>(acc);
      // sort using only 16 bits
      cms::alpakatools::radixSort<Acc1, float, 2>(acc, ptv2, sortInd, sws, nvFinal);
#else
      for (uint16_t i = 0; i < nvFinal; ++i)
        sortInd[i] = i;
      std::sort(sortInd, sortInd + nvFinal, [&](auto i, auto j) { return ptv2[i] < ptv2[j]; });
#endif
    }

    struct sortByPt2Kernel {
      template <typename T_Acc>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc, ZVertices* pdata, WorkSpace* pws) const {
        sortByPt2(acc, pdata, pws);
      }
    };

  }  // namespace gpuVertexFinder

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
