#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h

//#include <algorithm>
#include <cmath>
#include <cstdint>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"
#include "SYCLCore/printf.h"
#include "SYCLCore/radixSort.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __attribute__((always_inline)) void sortByPt2(ZVertices* pdata, WorkSpace* pws, sycl::nd_item<1> item) {
    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ ptt2 = ws.ptt2;
    uint32_t const& nvFinal = data.nvFinal;

    int32_t const* __restrict__ iv = ws.iv;
    float* __restrict__ ptv2 = data.ptv2;           // empty, will be filled here
    uint16_t* __restrict__ sortInd = data.sortInd;  // empty, will be filled in radixSort

#ifdef VERTEX_DEBUG
    if (item.get_local_id(0) == 0)
      printf("sorting %d vertices\n", nvFinal);
#endif

    if (nvFinal < 1)
      return;

    // fill indexing
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      data.idv[ws.itrk[i]] = iv[i];
    }

    // can be done asynchronoisly at the end of previous event
    for (auto i = item.get_local_id(0); i < nvFinal; i += item.get_local_range(0)) {
      ptv2[i] = 0;
    }
    sycl::group_barrier(item.get_group());

    // ptt2 is the pt of the track squared
    // ptv2 is the "pt of the vertex" (i.e. sum of the pt^2 of the tracks that belong to that vertex) squared
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] > 9990)
        continue;
      cms::sycltools::atomic_fetch_add<float>(&ptv2[iv[i]], ptt2[i]);
    }
    sycl::group_barrier(item.get_group());
    // now only the first "number of vertices" entries of ptv2 will be relevant
    // because iv[i] goes from 0 to the number of vertices, while i from 0 to the number of tracks
    // even though ptv2 has size nt(=number of tracks)

    if (1 == nvFinal) {
      if (item.get_local_id(0) == 0)
        sortInd[0] = 0;
      return;
    }
    auto swsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint16_t[1024]>(item.get_group());
    uint16_t* sws = (uint16_t*)swsbuff.get();
    // TODO_ a different sort for CPU
    radixSort<float, 2>(ptv2, sortInd, sws, nvFinal, item);
  }

  __attribute__((always_inline)) void sortByPt2CPU(ZVertices* pdata, WorkSpace* pws, sycl::nd_item<1> item) {
    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ ptt2 = ws.ptt2;
    uint32_t const& nvFinal = data.nvFinal;

    int32_t const* __restrict__ iv = ws.iv;
    float* __restrict__ ptv2 = data.ptv2;           // empty, will be filled here
    uint16_t* __restrict__ sortInd = data.sortInd;  // empty, will be filled in radixSort

#ifdef VERTEX_DEBUG
    if (item.get_local_id(0) == 0)
      printf("sorting %d vertices\n", nvFinal);
#endif

    if (nvFinal < 1)
      return;

    // fill indexing
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      data.idv[ws.itrk[i]] = iv[i];
    }

    // can be done asynchronoisly at the end of previous event
    for (auto i = item.get_local_id(0); i < nvFinal; i += item.get_local_range(0)) {
      ptv2[i] = 0;
    }
    sycl::group_barrier(item.get_group());

    // ptt2 is the pt of the track squared
    // ptv2 is the "pt of the vertex" (i.e. sum of the pt^2 of the tracks that belong to that vertex) squared
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] > 9990)
        continue;
      cms::sycltools::atomic_fetch_add<float>(&ptv2[iv[i]], ptt2[i]);
    }
    sycl::group_barrier(item.get_group());
    // now only the first "number of vertices" entries of ptv2 will be relevant
    // because iv[i] goes from 0 to the number of vertices, while i from 0 to the number of tracks
    // even though ptv2 has size nt(=number of tracks)

    if (1 == nvFinal) {
      if (item.get_local_id(0) == 0)
        sortInd[0] = 0;
      return;
    } else {
      for (uint32_t i = item.get_local_id(0); i < nvFinal; i += item.get_local_range(0)) {
        sortInd[i] = i;
      }
    }
  }

  void sortByPt2Kernel(ZVertices* pdata, WorkSpace* pws, sycl::nd_item<1> item) { sortByPt2(pdata, pws, item); }

  void sortByPt2CPUKernel(ZVertices* pdata, WorkSpace* pws, sycl::nd_item<1> item) { sortByPt2CPU(pdata, pws, item); }
}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
