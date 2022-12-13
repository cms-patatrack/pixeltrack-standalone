#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <execution>
#include <memory>
#include <ranges>

#include "CUDACore/HistoContainer.h"
#include "CUDACore/portableAtomicOp.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __forceinline__ void fitVertices(ZVertices* pdata,
                                   WorkSpace* pws,
                                   float chi2Max  // for outlier rejection
  ) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ zt = ws.zt;
    float const* __restrict__ ezt2 = ws.ezt2;
    float* __restrict__ zv = data.zv;
    float* __restrict__ wv = data.wv;
    float* __restrict__ chi2 = data.chi2;
    uint32_t& nvFinal = data.nvFinal;
    uint32_t& nvIntermediate = ws.nvIntermediate;

    int32_t* __restrict__ nn = data.ndof;
    int32_t* __restrict__ iv = ws.iv;

    assert(pdata);
    assert(zt);

    assert(nvFinal <= nvIntermediate);
    nvFinal = nvIntermediate;
    auto foundClusters = nvFinal;

    // zero
    std::fill(zv, zv + foundClusters, 0);
    std::fill(wv, wv + foundClusters, 0);
    std::fill(chi2, chi2 + foundClusters, 0);

    // only for test
    std::unique_ptr<int> noise_p{std::make_unique<int>(0)};
    int* noise{noise_p.get()};

    auto iter{std::views::iota(0U, nt)};
    // one vertex per thread
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto i) {
      // compute cluster location
      if (iv[i] > 9990) {
        if (verbose)
          cms::cuda::atomicAdd(noise, 1);
        return;
      }
      assert(iv[i] >= 0);
      assert(iv[i] < int(foundClusters));
      auto w = 1.f / ezt2[i];
      cms::cuda::atomicAdd(&zv[iv[i]], zt[i] * w);
      cms::cuda::atomicAdd(&wv[iv[i]], w);
    });

    // reuse nn
    auto iter_fc{std::views::iota(0U, foundClusters)};
    std::for_each(std::execution::par, std::ranges::cbegin(iter_fc), std::ranges::cend(iter_fc), [=](const auto i) {
      assert(wv[i] > 0.f);
      zv[i] /= wv[i];
      nn[i] = -1;  // ndof
    });

    // compute chi2
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto i) {
      if (iv[i] > 9990)
        return;

      auto c2 = zv[iv[i]] - zt[i];
      c2 *= c2 / ezt2[i];
      if (c2 > chi2Max) {
        iv[i] = 9999;
        return;
      }
      cms::cuda::atomicAdd(&chi2[iv[i]], c2);
      cms::cuda::atomicAdd(&nn[iv[i]], 1);
    });

    std::for_each(std::execution::par, std::ranges::cbegin(iter_fc), std::ranges::cend(iter_fc), [=](const auto i) {
      if (nn[i] > 0)
        wv[i] *= float(nn[i]) / chi2[i];
    });

    if (verbose)
      printf("found %d proto clusters ", foundClusters);
    if (verbose)
      printf("and %d noise\n", *noise);
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
