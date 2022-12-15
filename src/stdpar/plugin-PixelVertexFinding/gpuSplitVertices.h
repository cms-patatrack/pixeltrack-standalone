#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <ranges>

#include "CUDACore/HistoContainer.h"
#include "CUDACore/portableAtomicOp.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __forceinline__ void splitVertices(ZVertices* pdata, WorkSpace* pws, float maxChi2) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ zt = ws.zt;
    float const* __restrict__ ezt2 = ws.ezt2;
    float* __restrict__ zv = data.zv;
    float* __restrict__ wv = data.wv;
    float const* __restrict__ chi2 = data.chi2;
    uint32_t& nvFinal = data.nvFinal;

    int32_t const* __restrict__ nn = data.ndof;
    int32_t* __restrict__ iv = ws.iv;

    assert(pdata);
    assert(zt);
    constexpr int MAXTK = 512;
    auto it_sp{std::make_unique<uint32_t[]>(MAXTK)};  // track index
    auto it{it_sp.get()};
    auto zz_sp{std::make_unique<float[]>(MAXTK)};  // z pos
    auto zz{zz_sp.get()};
    auto newV_sp{std::make_unique<uint8_t[]>(MAXTK)};  // 0 or 1
    auto newV{newV_sp.get()};
    auto ww_sp{std::make_unique<float[]>(MAXTK)};  // z weight
    auto ww{ww_sp.get()};

    auto iter{std::views::iota(0U, nvFinal)};
    // one vertex per thread
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto kv) {
      if (nn[kv] < 4)
        return;
      if (chi2[kv] < maxChi2 * float(nn[kv]))
        return;

      assert(nn[kv] < MAXTK);
      if (nn[kv] >= MAXTK)
        return;  // too bad FIXME

      uint32_t nq{0};  // number of track for this vertex

      // copy to local
      for (auto k = 0; k < nt; ++k) {
        if (iv[k] == int(kv)) {
          auto old = cms::cuda::atomicInc(&nq, static_cast<uint32_t>(MAXTK));
          zz[old] = zt[k] - zv[kv];
          newV[old] = zz[old] < 0 ? 0 : 1;
          ww[old] = 1.f / ezt2[k];
          it[old] = k;
        }
      }

      float znew[2], wnew[2];  // the new vertices

      assert(int(nq) == nn[kv] + 1);

      int maxiter = 20;
      // kt-min....
      bool more = true;
      while (__syncthreads_or(more)) {
        more = false;
        znew[0] = 0;
        znew[1] = 0;
        wnew[0] = 0;
        wnew[1] = 0;
        for (auto k = 0; k < nq; ++k) {
          auto i = newV[k];
          znew[i] += zz[k] * ww[k];
          wnew[i] += ww[k];
        }
        znew[0] /= wnew[0];
        znew[1] /= wnew[1];
        for (auto k = 0; k < nq; ++k) {
          auto d0 = fabs(zz[k] - znew[0]);
          auto d1 = fabs(zz[k] - znew[1]);
          auto newer = d0 < d1 ? 0 : 1;
          more |= newer != newV[k];
          newV[k] = newer;
        }
        --maxiter;
        if (maxiter <= 0)
          more = false;
      }

      // avoid empty vertices
      if (0 == wnew[0] || 0 == wnew[1])
        return;

      // quality cut
      auto dist2 = (znew[0] - znew[1]) * (znew[0] - znew[1]);

      auto chi2Dist = dist2 / (1.f / wnew[0] + 1.f / wnew[1]);

      if (verbose && 0 == kv)
        printf("inter %d %f %f\n", 20 - maxiter, chi2Dist, dist2 * wv[kv]);

      if (chi2Dist < 4)
        return;

      // get a new global vertex
      uint32_t igv;
      // we need to get a ref to the pointee for nvc++ to accept the atomic_ref construction
      auto& ws_d = *pws;
      std::atomic_ref<uint32_t> ws_intermediate_atomic{ws_d.nvIntermediate};
      igv = ws_intermediate_atomic++;
      for (auto k = 0; k < nq; ++k) {
        if (1 == newV[k])
          iv[it[k]] = igv;
      }
    });  // loop on vertices
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
