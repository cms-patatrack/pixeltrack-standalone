#ifndef plugin_PixelVertexFinding_alpaka_gpuFitVertices_h
#define plugin_PixelVertexFinding_alpaka_gpuFitVertices_h

#include "AlpakaCore/HistoContainer.h"
#include "AlpakaCore/config.h"

#include "gpuVertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace gpuVertexFinder {

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void fitVertices(
        const TAcc& acc,
        ZVertices* pdata,
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

      ALPAKA_ASSERT_ACC(pdata);
      ALPAKA_ASSERT_ACC(zt);

      ALPAKA_ASSERT_ACC(nvFinal <= nvIntermediate);
      nvFinal = nvIntermediate;
      auto foundClusters = nvFinal;

      // zero
      cms::alpakatools::for_each_element_in_block_strided(acc, foundClusters, [&](uint32_t i) {
        zv[i] = 0;
        wv[i] = 0;
        chi2[i] = 0;
      });

      // only for test
      auto& noise = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      if (verbose && 0 == threadIdxLocal)
        noise = 0;

      alpaka::syncBlockThreads(acc);

      // compute cluster location
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        if (iv[i] > 9990) {
          if (verbose)
            alpaka::atomicAdd(acc, &noise, 1, alpaka::hierarchy::Threads{});
        } else {
          ALPAKA_ASSERT_ACC(iv[i] >= 0);
          ALPAKA_ASSERT_ACC(iv[i] < int(foundClusters));
          auto w = 1.f / ezt2[i];
          alpaka::atomicAdd(acc, &zv[iv[i]], zt[i] * w, alpaka::hierarchy::Blocks{});
          alpaka::atomicAdd(acc, &wv[iv[i]], w, alpaka::hierarchy::Blocks{});
        }
      });

      alpaka::syncBlockThreads(acc);
      // reuse nn
      cms::alpakatools::for_each_element_in_block_strided(acc, foundClusters, [&](uint32_t i) {
        ALPAKA_ASSERT_ACC(wv[i] > 0.f);
        zv[i] /= wv[i];
        nn[i] = -1;  // ndof
      });
      alpaka::syncBlockThreads(acc);

      // compute chi2
      cms::alpakatools::for_each_element_in_block_strided(acc, nt, [&](uint32_t i) {
        if (iv[i] <= 9990) {
          auto c2 = zv[iv[i]] - zt[i];
          c2 *= c2 / ezt2[i];
          if (c2 > chi2Max) {
            iv[i] = 9999;
          } else {
            alpaka::atomicAdd(acc, &chi2[iv[i]], c2, alpaka::hierarchy::Blocks{});
            alpaka::atomicAdd(acc, &nn[iv[i]], 1, alpaka::hierarchy::Blocks{});
          }
        }
      });
      alpaka::syncBlockThreads(acc);

      cms::alpakatools::for_each_element_in_block_strided(acc, foundClusters, [&](uint32_t i) {
        if (nn[i] > 0)
          wv[i] *= float(nn[i]) / chi2[i];
      });

      if (verbose && 0 == threadIdxLocal)
        printf("found %d proto clusters ", foundClusters);
      if (verbose && 0 == threadIdxLocal)
        printf("and %d noise\n", noise);
    }

    struct fitVerticesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    ZVertices* pdata,
                                    WorkSpace* pws,
                                    float chi2Max  // for outlier rejection
      ) const {
        fitVertices(acc, pdata, pws, chi2Max);
      }
    };

  }  // namespace gpuVertexFinder

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_PixelVertexFinding_alpaka_gpuFitVertices_h
