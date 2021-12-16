#include "gpuVertexFinder.h"
#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

#include "KokkosCore/hintLightWeight.h"
#include "KokkosCore/shared_ptr.h"
#include "KokkosCore/ViewHelpers.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuVertexFinder {
    KOKKOS_INLINE_FUNCTION void loadTracks(
        const Kokkos::View<const pixelTrack::TrackSoA, KokkosDeviceMemSpace, RestrictUnmanaged>& tracks,
        const Kokkos::View<ZVertexSoA, KokkosDeviceMemSpace, RestrictUnmanaged>& soa,
        const Kokkos::View<WorkSpace, KokkosDeviceMemSpace, RestrictUnmanaged>& ws,
        const float ptMin,
        const size_t idx) {
      auto nHits = tracks().nHits(idx);
      if (nHits == 0)
        return;  // this is a guard: maybe we need to move to nTracks...

      auto const& fit = tracks().stateAtBS;
      auto const* quality = tracks().qualityData();

      // initialize soa...
      soa().idv[idx] = -1;

      if (nHits < 4)
        return;  // no triplets
      if (quality[idx] != trackQuality::loose)
        return;

      auto pt = tracks().pt(idx);

      if (pt < ptMin)
        return;

      auto& data = ws();
      auto it = cms::kokkos::atomic_fetch_add(&data.ntrks, 1U);
      data.itrk[it] = idx;
      data.zt[it] = tracks().zip(idx);
      data.ezt2[it] = fit.covariance(idx)(14);
      data.ptt2[it] = pt * pt;
    }

// #define THREE_KERNELS
#ifndef THREE_KERNELS
    void vertexFinderOneKernel(
        const Kokkos::View<gpuVertexFinder::ZVertices, KokkosDeviceMemSpace, RestrictUnmanaged>& vdata,
        const Kokkos::View<gpuVertexFinder::WorkSpace, KokkosDeviceMemSpace, RestrictUnmanaged>& vws,
        const Kokkos::View<gpuVertexFinder::ZVertices,
                           typename cms::kokkos::MemSpaceTraits<KokkosDeviceMemSpace>::HostSpace,
                           RestrictUnmanaged>& hdata,
        int minT,       // min number of neighbours to be "seed"
        float eps,      // max absolute distance to cluster
        float errmax,   // max error to be "seed"
        float chi2max,  // max normalized distance to cluster,
        KokkosExecSpace const& execSpace,
        Kokkos::TeamPolicy<KokkosExecSpace> const& teamPolicy) {
      clusterTracksByDensityHost(vdata, vws, minT, eps, errmax, chi2max, execSpace, teamPolicy);

      Kokkos::parallel_for(
          "vertexFinderOneKernel",
          hintLightWeight(teamPolicy),
          KOKKOS_LAMBDA(Kokkos::TeamPolicy<KokkosExecSpace>::member_type const& teamMember) {
            // 4 bytes of shared memory required
            fitVertices(vdata, vws, 50., teamMember);
            teamMember.team_barrier();

            splitVertices(vdata, vws, 9.f, teamMember);
            teamMember.team_barrier();

            fitVertices(vdata, vws, 5000., teamMember);
            teamMember.team_barrier();
          });

      sortByPt2Host(vdata, vws, hdata, execSpace, teamPolicy);
    }
#else
    void vertexFinderKernel1(
        const Kokkos::View<gpuVertexFinder::ZVertices, KokkosDeviceMemSpace, RestrictUnmanaged>& vdata,
        const Kokkos::View<gpuVertexFinder::WorkSpace, KokkosDeviceMemSpace, RestrictUnmanaged>& vws,
        int minT,       // min number of neighbours to be "seed"
        float eps,      // max absolute distance to cluster
        float errmax,   // max error to be "seed"
        float chi2max,  // max normalized distance to cluster,
        KokkosExecSpace const& execSpace,
        Kokkos::TeamPolicy<KokkosExecSpace> const& teamPolicy) {
      clusterTracksByDensityHost(vdata, vws, minT, eps, errmax, chi2max, execSpace, teamPolicy);
      Kokkos::parallel_for(
          "fitVertices_vertexFinderKernel1",
          hintLightWeight(teamPolicy),
          KOKKOS_LAMBDA(Kokkos::TeamPolicy<KokkosExecSpace>::member_type const& teamMember) {
            // 4 bytes of shared memory required
            fitVertices(vdata, vws, 50., teamMember);
          });
    }

    void vertexFinderKernel2(
        const Kokkos::View<gpuVertexFinder::ZVertices, KokkosDeviceMemSpace, RestrictUnmanaged>& vdata,
        const Kokkos::View<gpuVertexFinder::WorkSpace, KokkosDeviceMemSpace, RestrictUnmanaged>& vws,
        const Kokkos::View<gpuVertexFinder::ZVertices, KokkosHostMemSpace, RestrictUnmanaged>& hdata,
        KokkosExecSpace const& execSpace,
        Kokkos::TeamPolicy<KokkosExecSpace> const& teamPolicy) {
      Kokkos::parallel_for(
          "fitVertices_vertexFinderKernel2",
          hintLightWeight(teamPolicy),
          KOKKOS_LAMBDA(Kokkos::TeamPolicy<KokkosExecSpace>::member_type const& teamMember) {
            // 4 bytes of shared memory required
            fitVertices(vdata, vws, 5000., teamMember);
          });

      sortByPt2Host(vdata, vws, hdata, execSpace, teamPolicy);
    }
#endif

    cms::kokkos::shared_ptr<ZVertexSoA, KokkosDeviceMemSpace> Producer::make(
        cms::kokkos::shared_ptr<pixelTrack::TrackSoA, KokkosDeviceMemSpace> const& tksoa_ptr,
        float ptMin,
        KokkosExecSpace const& execSpace) const {
      // std::cout << "producing Vertices on GPU" << std::endl;
      auto vertices_d_ptr = cms::kokkos::make_shared<ZVertexSoA, KokkosDeviceMemSpace>(execSpace);
      auto vertices_h_ptr = cms::kokkos::make_mirror_shared(vertices_d_ptr, execSpace);
      auto workspace_d_ptr = cms::kokkos::make_shared<WorkSpace, KokkosDeviceMemSpace>(execSpace);

      auto vertices_d = cms::kokkos::to_view(vertices_d_ptr);
      auto vertices_h = cms::kokkos::to_view(vertices_h_ptr);
      auto workspace_d = cms::kokkos::to_view(workspace_d_ptr);
      auto tksoa = cms::kokkos::to_view(tksoa_ptr);

      using TeamPolicy = Kokkos::TeamPolicy<KokkosExecSpace>;
      using MemberType = Kokkos::TeamPolicy<KokkosExecSpace>::member_type;

      Kokkos::parallel_for(
          "init", hintLightWeight(TeamPolicy(execSpace, 1, 1)), KOKKOS_LAMBDA(MemberType const& teamMember) {
            vertices_d().nvFinal = 0;
            workspace_d().ntrks = 0;
            workspace_d().nvIntermediate = 0;
          });
      Kokkos::parallel_for(
          "loadTracks",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, TkSoA::stride())),
          KOKKOS_LAMBDA(const size_t i) { loadTracks(tksoa, vertices_d, workspace_d, ptMin, i); });

#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
      auto policy = TeamPolicy(execSpace, 1, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(8192 * 4));
#else
      auto policy = TeamPolicy(execSpace, 1, 128).set_scratch_size(0, Kokkos::PerTeam(8192 * 4));
#endif

      if (oneKernel_) {
        // implemented only for density clustesrs
#ifndef THREE_KERNELS
        // TODO: scratch sizes may need to be adjusted?
        // scratch size is from the unit test (not sure yet if it works here), following comment is also from there
        //FIXME: small scratch pad size will result in runtime error "an illegal memory access was encountered". Current
        // oneKernel test will NOT pass probably due to the high demand of scratch memory from splitVertices kernel
        vertexFinderOneKernel(vertices_d, workspace_d, vertices_h, minT, eps, errmax, chi2max, execSpace, policy);
        Kokkos::deep_copy(execSpace, vertices_d, vertices_h);
#else
        vertexFinderKernel1(vertices_d, workspace_d, minT, eps, errmax, chi2max, execSpace, policy);
        // one block per vertex...
        Kokkos::parallel_for(
            "splitVertices" hintLightWeight(TeamPolicy(execSpace, 1024, 128).set_sctratch_size(8192 * 4)),
            KOKKOS_LAMBDA(MemberType const& teamMember) { splitVertices(vertices_d, workspace_d, 9.f, teamMember); });
        vertexFinderKernel2(vertices_d, workspace_d, vertices_h, execSpace, policy);
#endif
      } else {  // five kernels
        if (useDensity_) {
          clusterTracksByDensityHost(vertices_d, workspace_d, minT, eps, errmax, chi2max, execSpace, policy);
        } else if (useDBSCAN_) {
          clusterTracksDBSCANHost(vertices_d, workspace_d, minT, eps, errmax, chi2max, execSpace, policy);
        } else if (useIterative_) {
          clusterTracksIterativeHost(vertices_d, workspace_d, minT, eps, errmax, chi2max, execSpace, policy);
        }
        Kokkos::parallel_for(
            "fitVertices",
            hintLightWeight(policy),
            KOKKOS_LAMBDA(Kokkos::TeamPolicy<KokkosExecSpace>::member_type const& teamMember) {
              // 4 bytes of shared memory required
              fitVertices(vertices_d, workspace_d, 50., teamMember);
            });
        // one block per vertex...
        Kokkos::parallel_for(
            "splitVertices",
#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
            hintLightWeight(TeamPolicy(execSpace, 1024, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(8192 * 4))),
#else
            hintLightWeight(TeamPolicy(execSpace, 1024, 128).set_scratch_size(0, Kokkos::PerTeam(8192 * 4))),
#endif
            KOKKOS_LAMBDA(MemberType const& teamMember) { splitVertices(vertices_d, workspace_d, 9.f, teamMember); });
        Kokkos::parallel_for(
            "fitVertices",
            hintLightWeight(policy),
            KOKKOS_LAMBDA(Kokkos::TeamPolicy<KokkosExecSpace>::member_type const& teamMember) {
              // 4 bytes of shared memory required
              fitVertices(vertices_d, workspace_d, 5000., teamMember);
            });
        sortByPt2Host(vertices_d, workspace_d, vertices_h, execSpace, policy);
      }

      return vertices_d_ptr;
    }
  }  // namespace gpuVertexFinder
}  // namespace KOKKOS_NAMESPACE
