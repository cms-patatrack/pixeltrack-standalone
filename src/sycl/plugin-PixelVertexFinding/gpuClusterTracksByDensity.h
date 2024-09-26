#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/printf.h"
#include "SYCLCore/syclAtomic.h"

#include "gpuVertexFinder.h"

// #define GPU_DEBUG

namespace gpuVertexFinder {

  using Hist = cms::sycltools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;

  // this algo does not really scale as it works in a single block...
  // enough for <10K tracks we have
  //
  // based on Rodrighez&Laio algo
  //
  __attribute__((always_inline)) void clusterTracksByDensity(gpuVertexFinder::ZVertices* pdata,
                                                             gpuVertexFinder::WorkSpace* pws,
                                                             int minT,       // min number of neighbours to be "seed"
                                                             float eps,      // max absolute distance to cluster
                                                             float errmax,   // max error to be "seed"
                                                             float chi2max,  // max normalized distance to cluster
                                                             sycl::nd_item<1> item) {
    using namespace gpuVertexFinder;

#ifdef VERTEX_DEBUG
    if (item.get_local_id(0) == 0)
      printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);
#endif

    auto er2mx = errmax * errmax;

    auto& __restrict__ data = *pdata;          // info on tracks
    auto& __restrict__ ws = *pws;              // info on vertices
    auto nt = ws.ntrks;                        // number of tracks (uint32_t)
    float const* __restrict__ zt = ws.zt;      // z coord of the tracks at bs
    float const* __restrict__ ezt2 = ws.ezt2;  // squared error on the z coord

    uint32_t& nvFinal = data.nvFinal;              // final number of vertices
    uint32_t& nvIntermediate = ws.nvIntermediate;  // intermediate number of vertices

    uint8_t* __restrict__ izt = ws.izt;    // z coord of input tracks as an integer
    int32_t* __restrict__ nn = data.ndof;  // number of degrees of freedom / nearest neighbours of the vertices
    int32_t* __restrict__ iv = ws.iv;      // index of the vertex each track is associated to

    assert(pdata);
    assert(zt);

    auto hwsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<Hist::Counter[32]>(item.get_group());
    Hist::Counter* hws = (Hist::Counter*)hwsbuff.get();
    auto histbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<Hist>(item.get_group());
    Hist* hist = (Hist*)histbuff.get();

    for (auto j = item.get_local_id(0); j < Hist::totbins(); j += item.get_local_range(0)) {
      hist->off[j] = 0;
    }
    sycl::group_barrier(item.get_group());

#ifdef VERTEX_DEBUG
    if (item.get_local_id(0) == 0)
      printf("booked hist with %d bins, size %d for %d tracks\n", hist->nbins(), hist->capacity(), nt);
#endif

    assert(nt <= hist->capacity());

    // fill hist  (bin shall be wider than "eps")
    // here the z coord of each track is turned into an integer from 0 to 255
    // and used to increment the counts of the hist
    // iv[i] depend on the order tracks have been found and
    // the values are not the same if the program is executed multiple times
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      assert(i < ZVertices::MAXTRACKS);
      int iz = int(zt[i] * 10.);  // valid if eps<=0.1
      // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
      iz = std::min(std::max(iz, INT8_MIN), INT8_MAX);
      izt[i] = iz - INT8_MIN;
      assert(iz - INT8_MIN >= 0);
      assert(iz - INT8_MIN < 256);
      hist->count(izt[i]);
      iv[i] = i;
      nn[i] = 0;
    }
    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) < 32)
      hws[item.get_local_id(0)] = 0;  // used by prefix scan...
    sycl::group_barrier(item.get_group());
    hist->finalize(item, hws);
    sycl::group_barrier(item.get_group());
    assert(hist->size() == nt);
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      hist->fill(izt[i], uint16_t(i));
    }
    sycl::group_barrier(item.get_group());

    // count neighbours
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (ezt2[i] > er2mx)
        continue;
      auto loop = [&](uint32_t j) {
        if (i == j)
          return;
        auto dist = sycl::fabs(zt[i] - zt[j]);
        if (dist > eps)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;
        nn[i]++;
      };

      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
    }

    sycl::group_barrier(item.get_group());

    // find closest above me .... (we ignore the possibility of two j at same distance from i)
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      float mdist = eps;
      auto loop = [&](uint32_t j) {
        if (nn[j] < nn[i])
          return;
        if (nn[j] == nn[i] && zt[j] >= zt[i])
          return;  // if equal use natural order...
        auto dist = sycl::fabs(zt[i] - zt[j]);
        if (dist > mdist)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;  // (break natural order???)
        mdist = dist;
        iv[i] = j;  // assign to cluster (better be unique??)
      };
      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
    }

    sycl::group_barrier(item.get_group());

#ifdef GPU_DEBUG
    //  mini verification
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] != int(i))
        assert(iv[iv[i]] != int(i));
    }
    sycl::group_barrier(item.get_group());
#endif

    // consolidate graph (percolate index of seed)
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      auto m = iv[i];
      while (m != iv[m])
        m = iv[m];
      iv[i] = m;
    }

#ifdef GPU_DEBUG
    sycl::group_barrier(item.get_group());
    //  mini verification
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] != int(i))
        assert(iv[iv[i]] != int(i));
    }
#endif

#ifdef GPU_DEBUG
    // and verify that we did not spit any cluster...
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      auto minJ = i;
      auto mdist = eps;
      auto loop = [&](uint32_t j) {
        if (nn[j] < nn[i])
          return;
        if (nn[j] == nn[i] && zt[j] >= zt[i])
          return;  // if equal use natural order...
        auto dist = sycl::fabs(zt[i] - zt[j]);
        if (dist > mdist)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;
        mdist = dist;
        minJ = j;
      };
      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
      // should belong to the same cluster...
      assert(iv[i] == iv[minJ]);
      assert(nn[i] <= nn[iv[i]]);
    }
    sycl::group_barrier(item.get_group());
#endif

    auto foundClustersbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned int>(item.get_group());
    unsigned int* foundClusters = (unsigned int*)foundClustersbuff.get();
    *foundClusters = 0;
    sycl::group_barrier(item.get_group());

    // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
    // mark these tracks with a negative id.
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] == int(i)) {
        if (nn[i] >= minT) {
          auto old = cms::sycltools::atomic_fetch_add<unsigned int, sycl::access::address_space::local_space>(
              foundClusters, (unsigned int)1);
          iv[i] = -(old + 1);
        } else {  // noise
          iv[i] = -9998;
        }
      }
    }
    sycl::group_barrier(item.get_group());

    assert(*foundClusters < ZVertices::MAXVTX);

    // propagate the negative id to all the tracks in the cluster.
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] >= 0) {
        // mark each track in a cluster with the same id as the first one
        iv[i] = iv[iv[i]];
      }
    }
    sycl::group_barrier(item.get_group());

    // adjust the cluster id to be a positive value starting from 0
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      iv[i] = -iv[i] - 1;
    }

    nvIntermediate = nvFinal = *foundClusters;

#ifdef VERTEX_DEBUG
    if (item.get_local_id(0) == 0)
      printf("found %d proto vertices\n", *foundClusters);
#endif
  }

  void clusterTracksByDensityKernel(gpuVertexFinder::ZVertices* pdata,
                                    gpuVertexFinder::WorkSpace* pws,
                                    int minT,       // min number of neighbours to be "seed"
                                    float eps,      // max absolute distance to cluster
                                    float errmax,   // max error to be "seed"
                                    float chi2max,  // max normalized distance to cluster
                                    sycl::nd_item<1> item) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max, item);
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h
