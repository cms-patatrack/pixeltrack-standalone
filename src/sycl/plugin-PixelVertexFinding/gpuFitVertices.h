#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"
#include "SYCLCore/printf.h"

#include "gpuVertexFinder.h"

// #define VERTEX_DEBUG

namespace gpuVertexFinder {

  __attribute__((always_inline)) void fitVertices(ZVertices* pdata,
                                                  WorkSpace* pws,
                                                  float chi2Max,  // for outlier rejection
                                                  sycl::nd_item<1> item) {
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
    for (auto i = item.get_local_id(0); i < foundClusters; i += item.get_local_range(0)) {
      zv[i] = 0;
      wv[i] = 0;
      chi2[i] = 0;
    }

    // only for test
#ifdef VERTEX_DEBUG
    auto noisebuff = sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
    int* noise = (int*)noisebuff.get();
    if (0 == item.get_local_id(0))
      *noise = 0;
#endif
    sycl::group_barrier(item.get_group());

    // compute cluster location
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] > 9990) {
#ifdef VERTEX_DEBUG
        cms::sycltools::atomic_fetch_add<int, sycl::access::address_space::local_space>(noise, 1);
#endif
        continue;
      }
      assert(iv[i] >= 0);
      assert(iv[i] < int(foundClusters));
      auto w = 1.f / ezt2[i];
      cms::sycltools::atomic_fetch_add<float>(&zv[iv[i]], (float)(zt[i] * w));
      cms::sycltools::atomic_fetch_add<float>(&wv[iv[i]], (float)w);
    }

    sycl::group_barrier(item.get_group());
    // reuse nn
    for (auto i = item.get_local_id(0); i < foundClusters; i += item.get_local_range(0)) {
      assert(wv[i] > 0.f);
      zv[i] /= wv[i];
      nn[i] = -1;  // ndof
    }
    sycl::group_barrier(item.get_group());

    // compute chi2
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] > 9990)
        continue;

      auto c2 = zv[iv[i]] - zt[i];
      c2 *= c2 / ezt2[i];
      if (c2 > chi2Max) {
        iv[i] = 9999;
        continue;
      }
      cms::sycltools::atomic_fetch_add<float>(&chi2[iv[i]], (float)c2);
      cms::sycltools::atomic_fetch_add<int32_t>(&nn[iv[i]], (int32_t)1);
    }
    sycl::group_barrier(item.get_group());
    for (auto i = item.get_local_id(0); i < foundClusters; i += item.get_local_range(0))
      if (nn[i] > 0)
        wv[i] *= float(nn[i]) / chi2[i];

#ifdef VERTEX_DEBUG
    if (0 == item.get_local_id(0))
      printf("found %d proto clusters and %d noise\n", foundClusters, *noise);
#endif
  }

  void fitVerticesKernel(ZVertices* pdata,
                         WorkSpace* pws,
                         float chi2Max,  // for outlier rejection
                         sycl::nd_item<1> item) {
    fitVertices(pdata, pws, chi2Max, item);
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
