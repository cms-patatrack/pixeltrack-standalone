#include "SYCLCore/syclAtomic.h"

#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

// #define VERTEX_DEBUG
// #define GPU_DEBUG

/* NOTE: SYCL_BUG_
 * in SplitVertices and in GpuSortByPt2 (radixSort) there are any/all_of_group
 * to work on CPU they require a sub_group_size = numberOfThreadsPerBlock
 * set the latter to 32 if you want to run on CPU
 * Note that radixSort will not work in this case (it requires a group size of 256)
 * working on a different sorting if we are on CPU
 */

namespace gpuVertexFinder {

  using Hist = cms::sycltools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;

  void loadTracks(TkSoA const* ptracks, ZVertexSoA* soa, WorkSpace* pws, float ptMin, sycl::nd_item<1> item) {
    assert(ptracks);
    assert(soa);
    auto const& tracks = *ptracks;
    auto const& fit = tracks.stateAtBS;
    auto const* quality = tracks.qualityData();

    auto first = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    for (int idx = first, nt = TkSoA::stride(); idx < nt; idx += item.get_group_range(0) * item.get_local_range(0)) {
      auto nHits = tracks.nHits(idx);
      if (nHits == 0)
        break;  // this is a guard: maybe we need to move to nTracks...

      // initialize soa...
      soa->idv[idx] = -1;

      if (nHits < 4)
        continue;  // no triplets
      if (quality[idx] != trackQuality::loose)
        continue;

      auto pt = tracks.pt(idx);

      if (pt < ptMin)
        continue;

      auto& data = *pws;
      auto it = cms::sycltools::atomic_fetch_add<uint32_t>(&data.ntrks, (uint32_t)1);
      data.itrk[it] = idx;
      data.zt[it] = tracks.zip(idx);
      data.ezt2[it] = fit.covariance(idx)(14);
      data.ptt2[it] = pt * pt;
    }
  }

// #define THREE_KERNELS
#ifndef THREE_KERNELS
  void vertexFinderOneKernelCPU(gpuVertexFinder::ZVertices* pdata,
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
  }

  void vertexFinderOneKernelGPU(gpuVertexFinder::ZVertices* pdata,
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
#else
  void vertexFinderKernel1(gpuVertexFinder::ZVertices* pdata,
                           gpuVertexFinder::WorkSpace* pws,
                           int minT,      // min number of neighbours to be "seed"
                           float eps,     // max absolute distance to cluster
                           float errmax,  // max error to be "seed"
                           float chi2max  // max normalized distance to cluster,
                               sycl::nd_item<1> item) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max, item);
    sycl::group_barrier(item.get_group());
    fitVertices(pdata, pws, 50., item);
  }

  void vertexFinderKernel2(gpuVertexFinder::ZVertices* pdata, gpuVertexFinder::WorkSpace* pws, sycl::nd_item<1> item) {
    fitVertices(pdata, pws, 5000., item);
    sycl::group_barrier(item.get_group());
    sortByPt2(pdata, pws, item);
  }
#endif

  ZVertexHeterogeneous Producer::makeAsync(sycl::queue stream, TkSoA const* tksoa, float ptMin, bool isCpu) const {
#ifdef VERTEX_DEBUG
    std::cout << "producing Vertices on GPU" << std::endl;
#endif

    ZVertexHeterogeneous vertices(cms::sycltools::make_device_unique<ZVertexSoA>(stream));
    assert(tksoa);
    auto* soa = vertices.get();
    assert(soa);

    auto ws_d = cms::sycltools::make_device_unique<WorkSpace>(stream);

    stream.submit([&](sycl::handler& cgh) {
      auto soa_kernel = soa;
      auto ws_kernel = ws_d.get();
      cgh.parallel_for<class init_vertex_Kernel>(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                                                 [=](sycl::nd_item<1> item) { init(soa_kernel, ws_kernel); });
    });

#ifdef GPU_DEBUG
    stream.wait();
#endif

    auto blockSize = 128;
    auto numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler& cgh) {
      auto tksoa_kernel = tksoa;
      auto soa_kernel = soa;
      auto ws_kernel = ws_d.get();
      cgh.parallel_for<class loadTracks_Kernel>(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item) { loadTracks(tksoa_kernel, soa_kernel, ws_kernel, ptMin, item); });
    });

#ifdef GPU_DEBUG
    stream.wait();
#endif

    if (oneKernel_) {
      // implemented only for density clustesrs
#ifndef THREE_KERNELS
      if (isCpu) {
        numberOfBlocks = 1;
        blockSize = 32;  // SYCL_BUG_ on GPU every size is ok, on CPU only 32 (i.e. the sub group size) will work
        stream.submit([&](sycl::handler& cgh) {
          auto soa_kernel = soa;
          auto ws_kernel = ws_d.get();
          auto minT_kernel = minT;
          auto eps_kernel = eps;
          auto errmax_kernel = errmax;
          auto chi2max_kernel = chi2max;
          cgh.parallel_for<class vertexFinder_one_Kernel_CPU>(
              sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
              [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                vertexFinderOneKernelCPU(
                    soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
              });
        });
        stream.submit([&](sycl::handler& cgh) {
          auto pdata = soa;
          cgh.single_task([=]() {
            auto& __restrict__ data = *pdata;
            float* ptv2 = data.ptv2;
            uint16_t* sortInd = data.sortInd;
            uint32_t const& nvFinal = data.nvFinal;
            bool sorting = true;
            while (sorting) {
              sorting = false;
              for (uint32_t i = 0; i < (nvFinal - 1); i++) {
                if (ptv2[i] > ptv2[i + 1]) {
                  // sort ptv2
                  auto tmp = ptv2[i];
                  ptv2[i] = ptv2[i + 1];
                  ptv2[i + 1] = tmp;

                  // sort indexes
                  auto tmp2 = sortInd[i];
                  sortInd[i] = sortInd[i + 1];
                  sortInd[i + 1] = tmp2;

                  //sorting still ongoing
                  sorting = true;
                }
              }
            }
          });
        });
      } else {
        numberOfBlocks = 1;
        blockSize =
            1024 - 256;  // SYCL_BUG_ on GPU every size is ok, on CPU only 32 (i.e. the sub group size) will work
        stream.submit([&](sycl::handler& cgh) {
          auto soa_kernel = soa;
          auto ws_kernel = ws_d.get();
          auto minT_kernel = minT;
          auto eps_kernel = eps;
          auto errmax_kernel = errmax;
          auto chi2max_kernel = chi2max;
          cgh.parallel_for<class vertexFinder_one_Kernel_GPU>(
              sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
              [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                vertexFinderOneKernelGPU(
                    soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
              });
        });
      }
#ifdef GPU_DEBUG
      stream.wait();
#endif

#else
      numberOfBlocks = 1;
      blockSize = 1024 - 256;
      stream.submit([&](sycl::handler& cgh) {
        auto soa_kernel = soa;
        auto ws_kernel = ws_d.get();
        cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                         [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                           vertexFinderKernel1(soa_kernel, ws_kernel, minT, eps, errmax, chi2max, item);
                         });
      });

#ifdef GPU_DEBUG
      stream.wait();
#endif

      numberOfBlocks = 1;
      blockSize = 1024 - 256;
      stream.submit([&](sycl::handler& cgh) {
        auto soa_kernel = soa;
        auto ws_kernel = ws_d.get();
        cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                         [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                           splitVerticesKernel(soa_kernel, ws_kernel, 9.f, item);
                         });
      });

#ifdef GPU_DEBUG
      stream.wait();
#endif

      numberOfBlocks = 1;
      blockSize = 1024 - 256;
      stream.submit([&](sycl::handler& cgh) {
        auto soa_kernel = soa;
        auto ws_kernel = ws_d.get();
        cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                         [=](sycl::nd_item<1> item)
                             [[intel::reqd_sub_group_size(32)]] { vertexFinderKernel2(soa_kernel, ws_kernel, item); });
      });

#ifdef GPU_DEBUG
      stream.wait();
#endif

#endif
    } else {  // five kernels
      if (useDensity_) {
        numberOfBlocks = 1;
        blockSize = 1024 - 256;
        stream.submit([&](sycl::handler& cgh) {
          auto soa_kernel = soa;
          auto ws_kernel = ws_d.get();
          auto minT_kernel = minT;
          auto eps_kernel = eps;
          auto errmax_kernel = errmax;
          auto chi2max_kernel = chi2max;
          cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                           [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                             clusterTracksByDensityKernel(
                                 soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
                           });
        });

#ifdef GPU_DEBUG
        stream.wait();
#endif

      } else if (useDBSCAN_) {
        numberOfBlocks = 1;
        blockSize = 1024 - 256;
        stream.submit([&](sycl::handler& cgh) {
          auto soa_kernel = soa;
          auto ws_kernel = ws_d.get();
          auto minT_kernel = minT;
          auto eps_kernel = eps;
          auto errmax_kernel = errmax;
          auto chi2max_kernel = chi2max;
          cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                           [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                             clusterTracksDBSCAN(
                                 soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
                           });
        });

#ifdef GPU_DEBUG
        stream.wait();
#endif

      } else if (useIterative_) {
        numberOfBlocks = 1;
        blockSize = 1024 - 256;
        stream.submit([&](sycl::handler& cgh) {
          auto soa_kernel = soa;
          auto ws_kernel = ws_d.get();
          auto minT_kernel = minT;
          auto eps_kernel = eps;
          auto errmax_kernel = errmax;
          auto chi2max_kernel = chi2max;
          cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                           [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                             clusterTracksIterative(
                                 soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
                           });
        });
      }

#ifdef GPU_DEBUG
      stream.wait();
#endif

      numberOfBlocks = 1;
      blockSize = 1024 - 256;
      stream.submit([&](sycl::handler& cgh) {
        auto soa_kernel = soa;
        auto ws_kernel = ws_d.get();
        cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                         [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                           fitVerticesKernel(soa_kernel, ws_kernel, 50., item);
                         });
      });

#ifdef GPU_DEBUG
      stream.wait();
#endif

      // one block per vertex...
      numberOfBlocks = 1024;
      blockSize = 128;
      stream.submit([&](sycl::handler& cgh) {
        auto soa_kernel = soa;
        auto ws_kernel = ws_d.get();
        cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                         [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                           splitVerticesKernel(soa_kernel, ws_kernel, 9.f, item);
                         });
      });

#ifdef GPU_DEBUG
      stream.wait();
#endif

      numberOfBlocks = 1;
      blockSize = 1024 - 256;
      stream.submit([&](sycl::handler& cgh) {
        auto soa_kernel = soa;
        auto ws_kernel = ws_d.get();
        cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                         [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                           fitVerticesKernel(soa_kernel, ws_kernel, 5000., item);
                         });
      });

#ifdef GPU_DEBUG
      stream.wait();
#endif
      numberOfBlocks = 1;
      blockSize = 256;
      if (isCpu) {
        stream.submit([&](sycl::handler& cgh) {
          auto soa_kernel = soa;
          auto ws_kernel = ws_d.get();
          cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                           [=](sycl::nd_item<1> item)
                               [[intel::reqd_sub_group_size(32)]] { sortByPt2CPUKernel(soa_kernel, ws_kernel, item); });
        });
        stream.submit([&](sycl::handler& cgh) {
          auto pdata = soa;
          cgh.single_task([=]() {
            auto& __restrict__ data = *pdata;
            float* ptv2 = data.ptv2;
            uint16_t* sortInd = data.sortInd;
            uint32_t const& nvFinal = data.nvFinal;
            bool sorting = true;
            while (sorting) {
              sorting = false;
              for (uint32_t i = 0; i < (nvFinal - 1); i++) {
                if (ptv2[i] > ptv2[i + 1]) {
                  // sort ptv2
                  auto tmp = ptv2[i];
                  ptv2[i] = ptv2[i + 1];
                  ptv2[i + 1] = tmp;

                  // sort indexes
                  auto tmp2 = sortInd[i];
                  sortInd[i] = sortInd[i + 1];
                  sortInd[i + 1] = tmp2;

                  //sorting still ongoing
                  sorting = true;
                }
              }
            }
          });
        });
      } else {
        numberOfBlocks = 1;
        blockSize = 256;
        stream.submit([&](sycl::handler& cgh) {
          auto soa_kernel = soa;
          auto ws_kernel = ws_d.get();
          cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                           [=](sycl::nd_item<1> item)
                               [[intel::reqd_sub_group_size(32)]] { sortByPt2Kernel(soa_kernel, ws_kernel, item); });
        });
      }
    }
#ifdef GPU_DEBUG
    stream.wait();
#endif
#ifdef __SYCL_TARGET_INTEL_X86_64__
    // FIXME needed only on CPU ?
    stream.wait();
#endif

    return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
