#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "CUDACore/HistoContainer.h"
#include "CUDACore/cuda_assert.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  void fitVertices(ZVertices* pdata,
                   WorkSpace* pws,
                   float chi2Max  // for outlier rejection
  ) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    constexpr uint32_t MAXTRACKS = WorkSpace::MAXTRACKS;
    constexpr uint32_t MAXVTX = WorkSpace::MAXVTX;

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
    //printf("foundClusters = %d MAXVTX = %d\n",foundClusters,MAXVTX);

#pragma omp target enter data map(alloc:zv[:MAXVTX], wv[:MAXVTX], chi2[:MAXVTX], nn[:MAXTRACKS]) \
                       map(to:iv[:MAXTRACKS], ezt2[:MAXTRACKS], zt[:MAXTRACKS])

    // zero
//#pragma omp target teams distribute parallel for map(from:zv[:MAXVTX], wv[:MAXVTX], chi2[:MAXVTX])
#pragma omp target teams distribute parallel for
    for (uint32_t i = 0; i < foundClusters; i++) {
      zv[i] = 0;
      wv[i] = 0;
      chi2[i] = 0;
    }

    // only for test
    int noise;
    if (verbose)
      noise = 0;

// Do assertion checks on CPU
    for (uint32_t i = 0; i < nt; i++) {
      if (iv[i] > 9990)
        continue;
      assert(iv[i] >= 0);
      assert(iv[i] < int(foundClusters));
    }

    // compute cluster location
//#pragma omp target teams distribute parallel for map(to:iv[:MAXTRACKS],ezt2[:MAXTRACKS],zt[:MAXTRACKS]) map(tofrom:zv[:MAXVTX],wv[:MAXVTX])
#pragma omp target teams distribute parallel for
    for (uint32_t i = 0; i < nt; i++) {
      if (iv[i] > 9990) {
        //if (verbose)
        //  atomicAdd(&noise, 1);
        continue;
      }
      // Will fail at runtime - kernel argument misamtch - when targeting AMD using LLVM
      //assert(iv[i] >= 0);
      //assert(iv[i] < int(foundClusters));
      auto w = 1.f / ezt2[i];
#pragma omp atomic update
      zv[iv[i]] += zt[i] * w;
#pragma omp atomic update
      wv[iv[i]] +=  w;
    }

    // reuse nn
#pragma omp target teams distribute parallel for
    for (uint32_t i = 0; i < foundClusters; i++) {
       // If this assertion is included in this offloaded loop, it fails validation (for events 19,40,41,53,...)
       //  on an AMD target
      //assert(wv[i] > 0.f);
      zv[i] /= wv[i];
      nn[i] = -1;  // ndof
    }




    // compute chi2
// For an AMD target, adding this loop will result in an assertion gpuSplitVertices.h:66 in event 6
#pragma omp target teams distribute parallel for
    for (uint32_t i = 0; i < nt; i++) {
      if (iv[i] > 9990)
        continue;

      auto c2 = zv[iv[i]] - zt[i];
      c2 *= c2 / ezt2[i];
      if (c2 > chi2Max) {
        iv[i] = 9999;
        continue;
      }

#pragma omp atomic update
      chi2[iv[i]] += c2;

#pragma omp atomic update
      nn[iv[i]]++;
      //atomicAdd(&chi2[iv[i]], c2);
      //atomicAdd(&nn[iv[i]], 1);
    }
//#pragma omp target exit data map(from:zv[:MAXVTX], wv[:MAXVTX] , chi2[:MAXVTX], nn[:MAXTRACKS]) \
//                       map(delete:iv[:MAXTRACKS], ezt2[:MAXTRACKS], zt[:MAXTRACKS])

#pragma omp target teams distribute parallel for
    for (uint32_t i = 0; i < foundClusters; i++)
      if (nn[i] > 0)
        wv[i] *= float(nn[i]) / chi2[i];

#pragma omp target exit data map(from:zv[:MAXVTX], wv[:MAXVTX] , chi2[:MAXVTX], nn[:MAXTRACKS]) \
                       map(delete:iv[:MAXTRACKS], ezt2[:MAXTRACKS], zt[:MAXTRACKS])

    if (verbose)
      printf("found %d proto clusters ", foundClusters);
    if (verbose)
      printf("and %d noise\n", noise);
  }

  void fitVerticesKernel(ZVertices* pdata,
                         WorkSpace* pws,
                         float chi2Max  // for outlier rejection
  ) {
    fitVertices(pdata, pws, chi2Max);
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
