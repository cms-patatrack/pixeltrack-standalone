#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef TODO
#include "CUDACore/HistoContainer.h"
#include "CUDACore/cuda_assert.h"
#endif  // TODO

#include "gpuVertexFinder.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {
    // this algo does not really scale as it works in a single block...
    // enough for <10K tracks we have
    //
    // based on Rodrighez&Laio algo
    //
    KOKKOS_INLINE_FUNCTION void clusterTracksByDensity(
        Kokkos::View<ZVertices*, KokkosExecSpace> vdata,
        Kokkos::View<WorkSpace*, KokkosExecSpace> vws,
        int minT,       // min number of neighbours to be "seed"
        float eps,      // max absolute distance to cluster
        float errmax,   // max error to be "seed"
        float chi2max,  // max normalized distance to cluster
        const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& team_member) {
#ifdef TODO
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false

      auto id = team_member.league_rank() * team_member.team_size() + team_member.team_rank();

      if (verbose && 0 == id)
        printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);

      auto er2mx = errmax * errmax;

      auto& __restrict__ data = *vdata.data();
      auto& __restrict__ ws = *vws.data();

      auto nt = ws.ntrks;
      float const* __restrict__ zt = ws.zt;
      float const* __restrict__ ezt2 = ws.ezt2;

      uint32_t& nvFinal = data.nvFinal;
      uint32_t& nvIntermediate = ws.nvIntermediate;

      uint8_t* __restrict__ izt = ws.izt;
      int32_t* __restrict__ nn = data.ndof;
      int32_t* __restrict__ iv = ws.iv;

      assert(vdata.data());
      assert(zt);

      using Hist = HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;

      // Get shared team allocations on the scratch pad
      Hist* hist = (Hist*)team_member.team_shmem().get_shmem(sizeof(Hist));
      Hist::Counter* hws = (Hist::Counter*)team_member.team_shmem().get_shmem(sizeof(Hist::Counter) * 32);

      for (auto j = team_member.team_rank(); j < Hist::totbins(); j += team_member.team_size()) {
        hist->off[j] = 0;
      }

      team_member.team_barrier();

      if (verbose && 0 == id)
        printf("booked hist with %d bins, size %d for %d tracks\n", hist->nbins(), hist->capacity(), nt);

      assert(nt <= hist->capacity());

      // fill hist  (bin shall be wider than "eps")
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
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
      team_member.team_barrier();
      if (id < 32)
        hws[id] = 0;  // used by prefix scan...
      team_member.team_barrier();
      hist->finalize(hws);
      team_member.team_barrier();
      assert(hist->size() == nt);
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        hist->fill(izt[i], uint16_t(i));
      }
      team_member.team_barrier();

      // count neighbours
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        if (ezt2[i] > er2mx)
          continue;
        auto loop = [&](uint32_t j) {
          if (i == j)
            return;
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > eps)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;
          nn[i]++;
        };

        forEachInBins(hist, izt[i], 1, loop);
      }

      team_member.team_barrier();

      // find closest above me .... (we ignore the possibility of two j at same distance from i)
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        float mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < nn[i])
            return;
          if (nn[j] == nn[i] && zt[j] >= zt[i])
            return;  // if equal use natural order...
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;  // (break natural order???)
          mdist = dist;
          iv[i] = j;  // assign to cluster (better be unique??)
        };
        forEachInBins(hist, izt[i], 1, loop);
      }

      team_member.team_barrier();

#ifdef GPU_DEBUG
      //  mini verification
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        if (iv[i] != int(i))
          assert(iv[iv[i]] != int(i));
      }
      team_member.team_barrier();
#endif

      // consolidate graph (percolate index of seed)
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        auto m = iv[i];
        while (m != iv[m])
          m = iv[m];
        iv[i] = m;
      }

#ifdef GPU_DEBUG
      team_member.team_barrier();
      //  mini verification
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        if (iv[i] != int(i))
          assert(iv[iv[i]] != int(i));
      }
#endif

#ifdef GPU_DEBUG
      // and verify that we did not spit any cluster...
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        auto minJ = i;
        auto mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < nn[i])
            return;
          if (nn[j] == nn[i] && zt[j] >= zt[i])
            return;  // if equal use natural order...
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;
          mdist = dist;
          minJ = j;
        };
        forEachInBins(hist, izt[i], 1, loop);
        // should belong to the same cluster...
        assert(iv[i] == iv[minJ]);
        assert(nn[i] <= nn[iv[i]]);
      }
      team_member.team_barrier();
#endif

      unsigned int* foundClusters = (unsigned int*)team_member.team_shmem().get_shmem(sizeof(unsigned int));
      foundClusters[0] = 0;
      team_member.team_barrier();

      // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
      // mark these tracks with a negative id.
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        if (iv[i] == int(i)) {
          if (nn[i] >= minT) {
            auto old = Kokkos::atomic_increment(foundClusters);
            iv[i] = -(old + 1);
          } else {  // noise
            iv[i] = -9998;
          }
        }
      }
      team_member.team_barrier();

      assert(foundClusters[0] < ZVertices::MAXVTX);

      // propagate the negative id to all the tracks in the cluster.
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        if (iv[i] >= 0) {
          // mark each track in a cluster with the same id as the first one
          iv[i] = iv[iv[i]];
        }
      }
      team_member.team_barrier();

      // adjust the cluster id to be a positive value starting from 0
      for (auto i = team_member.team_rank(); i < nt; i += team_member.team_size()) {
        iv[i] = -iv[i] - 1;
      }

      nvIntermediate = nvFinal = foundClusters[0];

      if (verbose && 0 == id)
        printf("found %d proto vertices\n", foundClusters[0]);
#endif  // TODO
    }

    KOKKOS_INLINE_FUNCTION void clusterTracksByDensityKernel(gpuVertexFinder::ZVertices* pdata,
                                                             gpuVertexFinder::WorkSpace* pws,
                                                             int minT,      // min number of neighbours to be "seed"
                                                             float eps,     // max absolute distance to cluster
                                                             float errmax,  // max error to be "seed"
                                                             float chi2max  // max normalized distance to cluster
    ) {
      //clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
    }

  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h
