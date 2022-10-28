#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "CUDACore/HistoContainer.h"
#include "CUDACore/cuda_assert.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  inline void splitVertices(ZVertices* pdata, WorkSpace* pws, float maxChi2) {
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

    // one vertex per block
    for (uint32_t kv = 0; kv < nvFinal; kv += 1) {
      if (nn[kv] < 4)
        continue;
      if (chi2[kv] < maxChi2 * float(nn[kv]))
        continue;

      constexpr int MAXTK = 512;
      assert(nn[kv] < MAXTK);
      if (nn[kv] >= MAXTK)
        continue;           // too bad FIXME
      uint32_t it[MAXTK];   // track index
      float zz[MAXTK];      // z pos
      uint8_t newV[MAXTK];  // 0 or 1
      float ww[MAXTK];      // z weight

      uint32_t nq;  // number of track for this vertex
      nq = 0;

      // copy to local
      for (uint32_t k = 0; k < nt; k++) {
        if (iv[k] == int(kv)) {
          auto old = atomicInc(&nq, MAXTK);
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
      while (more) {
        more = false;

        znew[0] = 0;
        znew[1] = 0;
        wnew[0] = 0;
        wnew[1] = 0;

        for (uint32_t k = 0; k < nq; k++) {
          auto i = newV[k];
          atomicAdd(&znew[i], zz[k] * ww[k]);
          atomicAdd(&wnew[i], ww[k]);
        }

        znew[0] /= wnew[0];
        znew[1] /= wnew[1];

        for (uint32_t k = 0; k < nq; k++) {
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
        continue;

      // quality cut
      auto dist2 = (znew[0] - znew[1]) * (znew[0] - znew[1]);

      auto chi2Dist = dist2 / (1.f / wnew[0] + 1.f / wnew[1]);

      if (verbose)
        printf("inter %d %f %f\n", 20 - maxiter, chi2Dist, dist2 * wv[kv]);

      if (chi2Dist < 4)
        continue;

      // get a new global vertex
      uint32_t igv;
      igv = atomicAdd(&ws.nvIntermediate, 1);

      for (uint32_t k = 0; k < nq; k++) {
        if (1 == newV[k])
          iv[it[k]] = igv;
      }

    }  // loop on vertices
  }

  void splitVerticesKernel(ZVertices* pdata, WorkSpace* pws, float maxChi2) { splitVertices(pdata, pws, maxChi2); }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
