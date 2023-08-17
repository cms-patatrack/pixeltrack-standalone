#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "SYCLCore/chooseDevice.h"
#include "SYCLCore/printf.h"

#ifdef USE_DBSCAN
#include "plugin-PixelVertexFinding/gpuClusterTracksDBSCAN.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksDBSCAN
#elif USE_ITERATIVE
#include "plugin-PixelVertexFinding/gpuClusterTracksIterative.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksIterative
#else
#include "plugin-PixelVertexFinding/gpuClusterTracksByDensity.h"
#define CLUSTERIZE gpuVertexFinder::clusterTracksByDensityKernel
#endif
#include "plugin-PixelVertexFinding/gpuFitVertices.h"
#include "plugin-PixelVertexFinding/gpuSortByPt2.h"
#include "plugin-PixelVertexFinding/gpuSplitVertices.h"

#ifdef ONE_KERNEL
void vertexFinderOneKernel(gpuVertexFinder::ZVertices* pdata,
                           gpuVertexFinder::WorkSpace* pws,
                           int minT,       // min number of neighbours to be "seed"
                           float eps,      // max absolute distance to cluster
                           float errmax,   // max error to be "seed"
                           float chi2max,  // max normalized distance to cluster
                           sycl::nd_item<1> item) {
  clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max, item);
  sycl::group_barrier(item.get_group());
  fitVertices(pdata, pws, 50., item);
  sycl::group_barrier(item.get_group());
  splitVertices(pdata, pws, 9.f, item);
  sycl::group_barrier(item.get_group());
  fitVertices(pdata, pws, 5000., item);
  sycl::group_barrier(item.get_group());
  sortByPt2(pdata, pws, item);
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
#define LOC_ONGPU(M) ((char*)(onGPU_d.get()) + offsetof(gpuVertexFinder::ZVertices, M))
#define LOC_WS(M) ((char*)(ws_d.get()) + offsetof(gpuVertexFinder::WorkSpace, M))

void print(gpuVertexFinder::ZVertices const* pdata, gpuVertexFinder::WorkSpace const* pws) {
  auto const& __restrict__ data = *pdata;
  auto const& __restrict__ ws = *pws;
  printf("nt,nv %d %d,%d\n", ws.ntrks, data.nvFinal, ws.nvIntermediate);
}

int main(int argc, char** argv) {
  std::string devices(argv[1]);
  setenv("ONEAPI_DEVICE_SELECTOR", devices.c_str(), true);

  cms::sycltools::enumerateDevices(true);
  sycl::device device = cms::sycltools::chooseDevice(0);
  sycl::queue queue = sycl::queue(device, sycl::property::queue::in_order());

  std::cout << "VertexFinder offload to " << device.get_info<sycl::info::device::name>() << " on backend "
            << device.get_backend() << std::endl;

  auto onGPU_d = cms::sycltools::make_device_unique<gpuVertexFinder::ZVertices[]>(1, queue);
  auto ws_d = cms::sycltools::make_device_unique<gpuVertexFinder::WorkSpace[]>(1, queue);

  Event ev;

  float eps = 0.1f;
  std::array<float, 3> par{{eps, 0.01f, 9.0f}};
  for (int nav = 30; nav < 80; nav += 20) {
    ClusterGenerator gen(nav, 10);

    for (int i = 8; i < 20; ++i) {
      auto kk = i / 4;  // M param

      gen(ev);

      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        cgh.parallel_for<class init_vertex_Kernel_t>(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                                                     [=](sycl::nd_item<1> item) { init(soa_kernel, ws_kernel); });
      });

      std::cout << "v,t size " << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;
      auto nt = ev.ztrack.size();

      queue.memcpy(LOC_WS(ntrks), &nt, sizeof(uint32_t));
      queue.memcpy(LOC_WS(zt), ev.ztrack.data(), sizeof(float) * ev.ztrack.size());
      queue.memcpy(LOC_WS(ezt2), ev.eztrack.data(), sizeof(float) * ev.eztrack.size());
      queue.memcpy(LOC_WS(ptt2), ev.pttrack.data(), sizeof(float) * ev.eztrack.size());

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

      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        cgh.parallel_for<class print_Kernel_1>(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                                               [=](sycl::nd_item<1> item) { print(soa_kernel, ws_kernel); });
      });

      queue.wait_and_throw();

      auto numberOfBlocks = 1;
      auto blockSize = 512 + 256;
#ifdef ONE_KERNEL
      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        auto minT_kernel = kk;
        auto eps_kernel = par[0];
        auto errmax_kernel = par[1];
        auto chi2max_kernel = par[2];
        cgh.parallel_for<class vertexFinder_one_Kernel_t>(
            sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
              vertexFinderOneKernel(
                  soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
            });
      });
#else
      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        auto minT_kernel = kk;
        auto eps_kernel = par[0];
        auto errmax_kernel = par[1];
        auto chi2max_kernel = par[2];
        cgh.parallel_for<class clusterizer_Kernel_t>(
            sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
              CLUSTERIZE(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
            });
      });
#endif

      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        cgh.parallel_for<class print_Kernel_2>(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                                               [=](sycl::nd_item<1> item) { print(soa_kernel, ws_kernel); });
      });

      queue.wait_and_throw();

      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        cgh.parallel_for<class fitVertices_Kernel_t1>(
            sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item)
                [[intel::reqd_sub_group_size(32)]] { fitVerticesKernel(soa_kernel, ws_kernel, 50., item); });
      });

      queue.memcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t));

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

      float* zv = nullptr;
      float* wv = nullptr;
      float* ptv2 = nullptr;
      int32_t* nn = nullptr;
      uint16_t* ind = nullptr;

      // keep chi2 separated...
      float chi2[2 * nv];  // make space for splitting...

      float hzv[2 * nv];
      float hwv[2 * nv];
      float hptv2[2 * nv];
      int32_t hnn[2 * nv];
      uint16_t hind[2 * nv];

      zv = hzv;
      wv = hwv;
      ptv2 = hptv2;
      nn = hnn;
      ind = hind;

      queue.memcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t));
      queue.memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "after fit nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        cgh.parallel_for<class fitVertices_Kernel_t2>(
            sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item)
                [[intel::reqd_sub_group_size(32)]] { fitVerticesKernel(soa_kernel, ws_kernel, 50., item); });
      });
      queue.memcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t));
      queue.memcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t));
      queue.memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "before splitting nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      // one vertex per block!!!
      numberOfBlocks = 1024;
      blockSize = 64;
      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        cgh.parallel_for<class splitVertices_Kernel_t>(
            sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
              gpuVertexFinder::splitVerticesKernel(soa_kernel, ws_kernel, 9.f, item);
            });
      });
      queue.memcpy(&nv, LOC_WS(nvIntermediate), sizeof(uint32_t));

      std::cout << "after split " << nv << std::endl;
      numberOfBlocks = 1;
      blockSize = 1024 - 256;
      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        cgh.parallel_for<class fitVertices_Kernel_t3>(
            sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item)
                [[intel::reqd_sub_group_size(32)]] { fitVerticesKernel(soa_kernel, ws_kernel, 5000., item); });
      });

      numberOfBlocks = 1;
      blockSize = 256;
      queue.submit([&](sycl::handler& cgh) {
        auto soa_kernel = onGPU_d.get();
        auto ws_kernel = ws_d.get();
        blockSize = 256;
        cgh.parallel_for<class sortByPt2_Kernel_t>(
            sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item)
                [[intel::reqd_sub_group_size(32)]] { sortByPt2Kernel(soa_kernel, ws_kernel, item); });
      });

      queue.memcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t));

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

      queue.memcpy(zv, LOC_ONGPU(zv), nv * sizeof(float));
      queue.memcpy(wv, LOC_ONGPU(wv), nv * sizeof(float));
      queue.memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));
      queue.memcpy(ptv2, LOC_ONGPU(ptv2), nv * sizeof(float));
      queue.memcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t));
      queue.memcpy(ind, LOC_ONGPU(sortInd), nv * sizeof(uint16_t));
      queue.wait_and_throw();

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      {
        auto mx = std::minmax_element(wv, wv + nv);
        std::cout << "min max error " << 1. / std::sqrt(*mx.first) << ' ' << 1. / std::sqrt(*mx.second) << std::endl;
      }

      {
        auto mx = std::minmax_element(ptv2, ptv2 + nv);
        std::cout << "min max ptv2 " << *mx.first << ' ' << *mx.second << std::endl;
        std::cout << "min max ptv2 " << ptv2[ind[0]] << ' ' << ptv2[ind[nv - 1]] << " at " << ind[0] << ' '
                  << ind[nv - 1] << std::endl;
      }

      float dd[nv];
      for (auto kv = 0U; kv < nv; ++kv) {
        auto zr = zv[kv];
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

  return 0;
}
