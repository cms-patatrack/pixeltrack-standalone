#include <random>

#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

#ifdef USE_DBSCAN
#include "plugin-PixelVertexFinding/kokkos/gpuClusterTracksDBSCAN.h"
#define CLUSTERIZE clusterTracksDBSCANHost
#elif USE_ITERATIVE
#include "plugin-PixelVertexFinding/kokkos/gpuClusterTracksIterative.h"
#define CLUSTERIZE clusterTracksIterativeHost
#else
#include "plugin-PixelVertexFinding/kokkos/gpuClusterTracksByDensity.h"
#define CLUSTERIZE clusterTracksByDensityHost
#endif
#include "plugin-PixelVertexFinding/kokkos/gpuFitVertices.h"
#include "plugin-PixelVertexFinding/kokkos/gpuSortByPt2.h"
#include "plugin-PixelVertexFinding/kokkos/gpuSplitVertices.h"

using team_policy = Kokkos::TeamPolicy<KokkosExecSpace>;
using member_type = Kokkos::TeamPolicy<KokkosExecSpace>::member_type;

#ifdef ONE_KERNEL
void vertexFinderOneKernel(Kokkos::View<KOKKOS_NAMESPACE::gpuVertexFinder::ZVertices, KokkosExecSpace> vdata,
                           Kokkos::View<KOKKOS_NAMESPACE::gpuVertexFinder::WorkSpace, KokkosExecSpace> vws,
                           typename Kokkos::View<ZVertices, KokkosExecSpace>::HostMirror hdata,
                           int minT,       // min number of neighbours to be "seed"
                           float eps,      // max absolute distance to cluster
                           float errmax,   // max error to be "seed"
                           float chi2max,  // max normalized distance to cluster,
                           const team_policy& policy) {
  clusterTracksByDensityHost(vdata, vws, minT, eps, errmax, chi2max, KokkosExecSpace(), policy);

  Kokkos::parallel_for(
      "vertexFinderOneKernel", policy, KOKKOS_LAMBDA(const member_type& team_member) {
        // 4 bytes of shared memory required
        fitVertices(vdata, vws, 50., team_member);
        team_member.team_barrier();

        splitVertices(vdata, vws, 9.f, team_member);
        team_member.team_barrier();

        fitVertices(vdata, vws, 5000., team_member);
        team_member.team_barrier();
      });

  sortByPt2Host(vdata, vws, hdata, KokkosExecSpace(), policy);
}
#endif

struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t> itrack;
  std::vector<float> ztrack;
  std::vector<float> eztrack;
  std::vector<float> pttrack;
  std::vector<uint16_t> ivert;
};

struct ClusterGenerator {
  explicit ClusterGenerator(float nvert, float ntrack)
      : rgen(-13., 13), errgen(0.005, 0.025), clusGen(nvert), trackGen(ntrack), gauss(0., 1.), ptGen(1.) {}

  void operator()(Event& ev) {
    int nclus = clusGen(reng);
    ev.zvert.resize(nclus);
    ev.itrack.resize(nclus);
    for (auto& z : ev.zvert) {
      z = 3.5f * gauss(reng);
    }

    ev.ztrack.clear();
    ev.eztrack.clear();
    ev.ivert.clear();
    for (int iv = 0; iv < nclus; ++iv) {
      auto nt = trackGen(reng);
      ev.itrack[nclus] = nt;
      for (int it = 0; it < nt; ++it) {
        auto err = errgen(reng);  // reality is not flat....
        ev.ztrack.push_back(ev.zvert[iv] + err * gauss(reng));
        ev.eztrack.push_back(err * err);
        ev.ivert.push_back(iv);
        ev.pttrack.push_back((iv == 5 ? 1.f : 0.5f) + ptGen(reng));
        ev.pttrack.back() *= ev.pttrack.back();
      }
    }
    // add noise
    auto nt = 2 * trackGen(reng);
    for (int it = 0; it < nt; ++it) {
      auto err = 0.03f;
      ev.ztrack.push_back(rgen(reng));
      ev.eztrack.push_back(err * err);
      ev.ivert.push_back(9999);
      ev.pttrack.push_back(0.5f + ptGen(reng));
      ev.pttrack.back() *= ev.pttrack.back();
    }
  }

  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen;
  std::uniform_real_distribution<float> errgen;
  std::poisson_distribution<int> clusGen;
  std::poisson_distribution<int> trackGen;
  std::normal_distribution<float> gauss;
  std::exponential_distribution<float> ptGen;
};

// a macro SORRY
#define LOC_WS(M) ((char*)(ws_h.data()) + offsetof(KOKKOS_NAMESPACE::gpuVertexFinder::WorkSpace, M))

void test() {
  Kokkos::View<KOKKOS_NAMESPACE::gpuVertexFinder::ZVertices, KokkosExecSpace> onGPU_d("onGPU_d");
  Kokkos::View<KOKKOS_NAMESPACE::gpuVertexFinder::WorkSpace, KokkosExecSpace> ws_d("ws_d");

  Event ev;

  float eps = 0.1f;
  std::array<float, 3> par{{eps, 0.01f, 9.0f}};
  for (int nav = 30; nav < 80; nav += 20) {
    ClusterGenerator gen(nav, 10);

    for (int i = 8; i < 20; ++i) {
      auto kk = i / 4;  // M param

      gen(ev);

      Kokkos::parallel_for(
          "init", team_policy(KokkosExecSpace(), 1, 1), KOKKOS_LAMBDA(const member_type& team_member) {
            onGPU_d.data()->nvFinal = 0;
            ws_d.data()->ntrks = 0;
            ws_d.data()->nvIntermediate = 0;
          });

      std::cout << "v,t size " << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;
      auto nt = ev.ztrack.size();

      auto ws_h = Kokkos::create_mirror_view(ws_d);
      auto onGPU_h = Kokkos::create_mirror_view(onGPU_d);

      // copy data to host mirror
      ::memcpy(LOC_WS(ntrks), &nt, sizeof(uint32_t));
      ::memcpy(LOC_WS(zt), ev.ztrack.data(), sizeof(float) * ev.ztrack.size());
      ::memcpy(LOC_WS(ezt2), ev.eztrack.data(), sizeof(float) * ev.eztrack.size());
      ::memcpy(LOC_WS(ptt2), ev.pttrack.data(), sizeof(float) * ev.eztrack.size());
      // deep copy from host to device
      Kokkos::deep_copy(KokkosExecSpace(), ws_d, ws_h);

      std::cout << "M eps, pset " << kk << ' ' << eps << ' ' << (i % 4) << std::endl;

      if ((i % 4) == 0)
        par = {{eps, 0.02f, 12.0f}};
      if ((i % 4) == 1)
        par = {{eps, 0.02f, 9.0f}};
      if ((i % 4) == 2)
        par = {{eps, 0.01f, 9.0f}};
      if ((i % 4) == 3)
        par = {{0.7f * eps, 0.01f, 9.0f}};

      uint32_t nv = 0;
      Kokkos::parallel_for(
          "print", team_policy(KokkosExecSpace(), 1, 1), KOKKOS_LAMBDA(const member_type& team_member) {
            printf("nt,nv %d %d,%d\n", ws_d.data()->ntrks, onGPU_d.data()->nvFinal, ws_d.data()->nvIntermediate);
          });

      KokkosExecSpace().fence();

#ifdef ONE_KERNEL
      //FIXME: small scratch pad size will result in runtime error "an illegal memory access was encountered". Current
      // oneKernel test will NOT pass probably due to the high demand of scratch memory from splitVertices kernel
      team_policy policy = team_policy(KokkosExecSpace(), 1, 128).set_scratch_size(0, Kokkos::PerTeam(8192 * 4));
      vertexFinderOneKernel(onGPU_d, ws_d, onGPU_h, kk, par[0], par[1], par[2], policy);
      Kokkos::deep_copy(KokkosExecSpace(), onGPU_d, onGPU_h);
#else
      team_policy policy = team_policy(KokkosExecSpace(), 1, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(1024));
      CLUSTERIZE(onGPU_d, ws_d, kk, par[0], par[1], par[2], KokkosExecSpace(), policy);
#endif
      Kokkos::parallel_for(
          "print", team_policy(KokkosExecSpace(), 1, 1), KOKKOS_LAMBDA(const member_type& team_member) {
            printf("nt,nv %d %d,%d\n", ws_d.data()->ntrks, onGPU_d.data()->nvFinal, ws_d.data()->nvIntermediate);
          });

      KokkosExecSpace().fence();

      Kokkos::parallel_for(
          "fitVerticesKernel", policy, KOKKOS_LAMBDA(const member_type& team_member) {
            fitVerticesKernel(onGPU_d, ws_d, 50.f, team_member);
          });
      // deep copy from device to host
      Kokkos::deep_copy(KokkosExecSpace(), onGPU_h, onGPU_d);
      KokkosExecSpace().fence();

      nv = onGPU_h.data()->nvFinal;

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

      for (auto j = 0U; j < nv; ++j)
        if (onGPU_h.data()->ndof[j] > 0)
          onGPU_h.data()->chi2[j] /= float(onGPU_h.data()->ndof[j]);
      {
        auto mx = std::minmax_element(&(onGPU_h.data()->chi2[0]), &(onGPU_h.data()->chi2[nv]));
        std::cout << "after fit nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      Kokkos::parallel_for(
          "fitVerticesKernel", policy, KOKKOS_LAMBDA(const member_type& team_member) {
            fitVerticesKernel(onGPU_d, ws_d, 50.f, team_member);
          });
      Kokkos::deep_copy(KokkosExecSpace(), onGPU_h, onGPU_d);
      KokkosExecSpace().fence();

      nv = onGPU_h.data()->nvFinal;

      for (auto j = 0U; j < nv; ++j)
        if (onGPU_h.data()->ndof[j] > 0)
          onGPU_h.data()->chi2[j] /= float(onGPU_h.data()->ndof[j]);
      {
        auto mx = std::minmax_element(&(onGPU_h.data()->chi2[0]), &(onGPU_h.data()->chi2[nv]));
        std::cout << "before splitting nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      // 6744 bytes of scratch memory required for splitVerticesKernel with team_size = 64
#ifdef KOKKOS_BACKEND_CUDA
      policy = team_policy(KokkosExecSpace(), 1024, 64).set_scratch_size(0, Kokkos::PerTeam(8192));
#else
      policy = team_policy(KokkosExecSpace(), 1, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(8192));
#endif
      Kokkos::parallel_for(
          "splitVerticesKernel", policy, KOKKOS_LAMBDA(const member_type& team_member) {
            splitVerticesKernel(onGPU_d, ws_d, 9.f, team_member);
          });
      Kokkos::deep_copy(KokkosExecSpace(), ws_h, ws_d);
      nv = ws_h.data()->nvIntermediate;

      std::cout << "after split " << nv << std::endl;

      policy = team_policy(KokkosExecSpace(), 1, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(1024));
      Kokkos::parallel_for(
          "fitVerticesKernel", policy, KOKKOS_LAMBDA(const member_type& team_member) {
            fitVerticesKernel(onGPU_d, ws_d, 5000.f, team_member);
          });

      // equivalent to sortByPt2Kernel + deep copy to host
      sortByPt2Host(onGPU_d, ws_d, onGPU_h, KokkosExecSpace(), policy);
      nv = onGPU_h.data()->nvFinal;

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

      for (auto j = 0U; j < nv; ++j)
        if (onGPU_h.data()->ndof[j] > 0)
          onGPU_h.data()->chi2[j] /= float(onGPU_h.data()->ndof[j]);
      {
        auto mx = std::minmax_element(&(onGPU_h.data()->chi2[0]), &(onGPU_h.data()->chi2[nv]));
        std::cout << "nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      {
        auto mx = std::minmax_element(&(onGPU_h.data()->wv[0]), &(onGPU_h.data()->wv[nv]));
        std::cout << "min max error " << 1. / std::sqrt(*mx.first) << ' ' << 1. / std::sqrt(*mx.second) << std::endl;
      }

      {
        auto mx = std::minmax_element(&(onGPU_h.data()->ptv2[0]), &(onGPU_h.data()->ptv2[nv]));
        std::cout << "min max ptv2 " << *mx.first << ' ' << *mx.second << std::endl;
        auto ind_0 = onGPU_h.data()->sortInd[0];
        auto ind_nvminus1 = onGPU_h.data()->sortInd[nv - 1];
        std::cout << "min max ptv2 " << onGPU_h.data()->ptv2[ind_0] << ' ' << onGPU_h.data()->ptv2[ind_nvminus1]
                  << " at " << ind_0 << ' ' << ind_nvminus1 << std::endl;
      }

      float dd[nv];
      for (auto kv = 0U; kv < nv; ++kv) {
        auto zr = onGPU_h.data()->zv[kv];
        auto md = 500.0f;
        for (auto zs : ev.ztrack) {
          auto d = std::abs(zr - zs);
          md = std::min(d, md);
        }
        dd[kv] = md;
      }
      if (i == 6) {
        for (auto d : dd)
          std::cout << d << ' ';
        std::cout << std::endl;
      }
      auto mx = std::minmax_element(dd, dd + nv);
      float rms = 0;
      for (auto d : dd)
        rms += d * d;
      rms = std::sqrt(rms) / (nv - 1);
      std::cout << "min max rms " << *mx.first << ' ' << *mx.second << ' ' << rms << std::endl;

    }  // loop on events
  }    // lopp on ave vert
}

int main(void) {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  test();
  return 0;
}
