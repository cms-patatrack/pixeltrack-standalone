#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h

#include <algorithm>
#include <atomic>
#include <execution>
#include <ranges>
#include <cmath>
#include <cstdint>

#include "CUDACore/HistoContainer.h"
#include "CUDACore/portableAtomicOp.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  void sortByPt2(ZVertices* pdata, WorkSpace* pws) {
    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ ptt2 = ws.ptt2;
    uint32_t const& nvFinal = data.nvFinal;

    int32_t const* __restrict__ iv = ws.iv;
    float* __restrict__ ptv2 = data.ptv2;
    uint16_t* __restrict__ sortInd = data.sortInd;

    if (nvFinal < 1)
      return;
    auto iter_nt{std::views::iota(static_cast<uint32_t>(0), nt)};
    // fill indexing
    int16_t* idv = data.idv;
    uint16_t* itrk = ws.itrk;
    std::for_each(std::execution::par, std::ranges::cbegin(iter_nt), std::ranges::cend(iter_nt), [=](const auto i) {
      idv[itrk[i]] = iv[i];
    });

    std::fill(std::execution::par, ptv2, ptv2 + nvFinal, 0);
    std::for_each(std::execution::par, std::ranges::cbegin(iter_nt), std::ranges::cend(iter_nt), [=](const auto i) {
      if (iv[i] <= 9990) {
        cms::cuda::atomicAdd(&ptv2[iv[i]], ptt2[i]);
      }
    });

    if (1 == nvFinal) {
      sortInd[0] = 0;
      return;
    }
    auto iter_nv{std::views::iota(static_cast<uint32_t>(0), nvFinal)};
    std::for_each(std::execution::par, std::ranges::cbegin(iter_nv), std::ranges::cend(iter_nv), [=](const auto i) {
      sortInd[i] = i;
    });
    std::sort(std::execution::par, sortInd, sortInd + nvFinal, [=](const auto i, const auto j) -> bool {
      return ptv2[i] < ptv2[j];
    });
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
