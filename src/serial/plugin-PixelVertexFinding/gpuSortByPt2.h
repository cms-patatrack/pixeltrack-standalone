#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "CUDACore/HistoContainer.h"
#include "CUDACore/cuda_assert.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __device__ __forceinline__ void sortByPt2(ZVertices* pdata, WorkSpace* pws) {
    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ ptt2 = ws.ptt2;
    uint32_t const& nvFinal = data.nvFinal;

    int32_t const* __restrict__ iv = ws.iv;
    float* __restrict__ ptv2 = data.ptv2;
    uint16_t* __restrict__ sortInd = data.sortInd;

    // if (threadIdx.x == 0)
    //    printf("sorting %d vertices\n",nvFinal);

    if (nvFinal < 1)
      return;

    // fill indexing
    for (uint32_t i = 0; i < nt; i += blockDim.x) {
      data.idv[ws.itrk[i]] = iv[i];
    }

    // can be done asynchronoisly at the end of previous event
    for (uint32_t i = 0; i < nvFinal; i += blockDim.x) {
      ptv2[i] = 0;
    }
    __syncthreads();

    for (uint32_t i = 0; i < nt; i += blockDim.x) {
      if (iv[i] > 9990)
        continue;
      atomicAdd(&ptv2[iv[i]], ptt2[i]);
    }
    __syncthreads();

    if (1 == nvFinal) {
      if (true)
        sortInd[0] = 0;
      return;
    }
    for (uint16_t i = 0; i < nvFinal; ++i)
      sortInd[i] = i;
    std::sort(sortInd, sortInd + nvFinal, [&](auto i, auto j) { return ptv2[i] < ptv2[j]; });
  }

  __global__ void sortByPt2Kernel(ZVertices* pdata, WorkSpace* pws) { sortByPt2(pdata, pws); }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
