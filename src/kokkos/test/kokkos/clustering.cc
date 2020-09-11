#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

// dirty, but works
#include "plugin-SiPixelClusterizer/kokkos/gpuClustering.h"
#include "plugin-SiPixelClusterizer/kokkos/gpuClusterChargeCut.h"

void test() {
  using namespace gpuClustering;

  int numElements = 256 * 2000;
  // these in reality are already on GPU

  Kokkos::View<uint16_t*, KokkosExecSpace> d_id("d_id", numElements);
  auto h_id = Kokkos::create_mirror_view(d_id);
  Kokkos::View<uint16_t*, KokkosExecSpace> d_x("d_x", numElements);
  auto h_x = Kokkos::create_mirror_view(d_x);
  Kokkos::View<uint16_t*, KokkosExecSpace> d_y("d_y", numElements);
  auto h_y = Kokkos::create_mirror_view(d_y);
  Kokkos::View<uint16_t*, KokkosExecSpace> d_adc("d_adc", numElements);
  auto h_adc = Kokkos::create_mirror_view(d_adc);

  Kokkos::View<int*, KokkosExecSpace> d_clus("d_clus", numElements);
  auto h_clus = Kokkos::create_mirror_view(d_clus);

  Kokkos::parallel_for("var_init1",Kokkos::RangePolicy<KokkosExecSpace>(0,numElements),KOKKOS_LAMBDA(const size_t i){
      d_clus(i) = 0;
      d_id(i) = 0;
      d_x(i) = 0;
      d_y(i) = 0;
      d_adc(i) = 0;
    });

  Kokkos::deep_copy(KokkosExecSpace(),h_clus,d_clus);
  Kokkos::deep_copy(KokkosExecSpace(),h_id,d_id);
  Kokkos::deep_copy(KokkosExecSpace(),h_x,d_x);
  Kokkos::deep_copy(KokkosExecSpace(),h_y,d_y);
  Kokkos::deep_copy(KokkosExecSpace(),h_adc,d_adc);

  Kokkos::View<uint32_t*, KokkosExecSpace> d_moduleStart("d_moduleStart", MaxNumModules + 1);
  auto h_moduleStart = Kokkos::create_mirror_view(d_moduleStart);
  Kokkos::View<uint32_t*, KokkosExecSpace> d_clusInModule("d_clusInModule", MaxNumModules);
  auto h_clusInModule = Kokkos::create_mirror_view(d_clusInModule);
  Kokkos::View<uint32_t*, KokkosExecSpace> d_moduleId("d_moduleId", MaxNumModules);
  auto h_moduleId = Kokkos::create_mirror_view(d_moduleId);

  Kokkos::parallel_for("var_init2",Kokkos::RangePolicy<KokkosExecSpace>(0,MaxNumModules+1),KOKKOS_LAMBDA(const size_t i){
      d_moduleStart(i) = 0;
      if(i<MaxNumModules){
        d_moduleId(i) = 0;
        d_clusInModule(i) = 0;
      }
    });

  Kokkos::deep_copy(KokkosExecSpace(),h_moduleStart,d_moduleStart);
  Kokkos::deep_copy(KokkosExecSpace(),h_moduleId,d_moduleId);
  Kokkos::deep_copy(KokkosExecSpace(),h_clusInModule,d_clusInModule);


  // later random number
  int n = 0;
  int ncl = 0;
  int y[10] = {5, 7, 9, 1, 3, 0, 4, 8, 2, 6};

  auto generateClusters = [&](int kn) {
    auto addBigNoise = 1 == kn % 2;
    if (addBigNoise) {
      constexpr int MaxPixels = 1000;
      int id = 666;
      for (int x = 0; x < 140; x += 3) {
        for (int yy = 0; yy < 400; yy += 3) {
          h_id[n] = id;
          h_x[n] = x;
          h_y[n] = yy;
          h_adc[n] = 1000;
          ++n;
          ++ncl;
          if (MaxPixels <= ncl)
            break;
        }
        if (MaxPixels <= ncl)
          break;
      }
    }

    {
      // isolated
      int id = 42;
      int x = 10;
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = x;
      h_adc[n] = kn == 0 ? 100 : 5000;
      ++n;

      // first column
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = 0;
      h_adc[n] = 5000;
      ++n;
      // first columns
      ++ncl;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 2;
      h_adc[n] = 5000;
      ++n;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 1;
      h_adc[n] = 5000;
      ++n;

      // last column
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = 415;
      h_adc[n] = 5000;
      ++n;
      // last columns
      ++ncl;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 415;
      h_adc[n] = 2500;
      ++n;
      h_id[n] = id;
      h_x[n] = x + 80;
      h_y[n] = 414;
      h_adc[n] = 2500;
      ++n;

      // diagonal
      ++ncl;
      for (int x = 20; x < 25; ++x) {
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = x;
        h_adc[n] = 1000;
        ++n;
      }
      ++ncl;
      // reversed
      for (int x = 45; x > 40; --x) {
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = x;
        h_adc[n] = 1000;
        ++n;
      }
      ++ncl;
      h_id[n++] = InvId;  // error
      // messy
      int xx[5] = {21, 25, 23, 24, 22};
      for (int k = 0; k < 5; ++k) {
        h_id[n] = id;
        h_x[n] = xx[k];
        h_y[n] = 20 + xx[k];
        h_adc[n] = 1000;
        ++n;
      }
      // holes
      ++ncl;
      for (int k = 0; k < 5; ++k) {
        h_id[n] = id;
        h_x[n] = xx[k];
        h_y[n] = 100;
        h_adc[n] = kn == 2 ? 100 : 1000;
        ++n;
        if (xx[k] % 2 == 0) {
          h_id[n] = id;
          h_x[n] = xx[k];
          h_y[n] = 101;
          h_adc[n] = 1000;
          ++n;
        }
      }
    }
    {
      // id == 0 (make sure it works!
      int id = 0;
      int x = 10;
      ++ncl;
      h_id[n] = id;
      h_x[n] = x;
      h_y[n] = x;
      h_adc[n] = 5000;
      ++n;
    }
    // all odd id
    for (int id = 11; id <= 1800; id += 2) {
      if ((id / 20) % 2)
        h_id[n++] = InvId;  // error
      for (int x = 0; x < 40; x += 4) {
        ++ncl;
        if ((id / 10) % 2) {
          for (int k = 0; k < 10; ++k) {
            h_id[n] = id;
            h_x[n] = x;
            h_y[n] = x + y[k];
            h_adc[n] = 100;
            ++n;
            h_id[n] = id;
            h_x[n] = x + 1;
            h_y[n] = x + y[k] + 2;
            h_adc[n] = 1000;
            ++n;
          }
        } else {
          for (int k = 0; k < 10; ++k) {
            h_id[n] = id;
            h_x[n] = x;
            h_y[n] = x + y[9 - k];
            h_adc[n] = kn == 2 ? 10 : 1000;
            ++n;
            if (y[k] == 3)
              continue;  // hole
            if (id == 51) {
              h_id[n++] = InvId;
              h_id[n++] = InvId;
            }  // error
            h_id[n] = id;
            h_x[n] = x + 1;
            h_y[n] = x + y[k] + 2;
            h_adc[n] = kn == 2 ? 10 : 1000;
            ++n;
          }
        }
      }
    }
  };  // end lambda
  Kokkos::deep_copy(KokkosExecSpace(),d_clusInModule,h_clusInModule);
  Kokkos::deep_copy(KokkosExecSpace(),d_moduleId,h_moduleId);
  Kokkos::deep_copy(KokkosExecSpace(),d_clus,h_clus);

  for (auto kkk = 0; kkk < 5; ++kkk) {
    n = 0;
    ncl = 0;
    generateClusters(kkk);

    std::cout << "created " << n << " digis in " << ncl << " clusters" << std::endl;
    assert(n <= numElements);

    h_moduleStart(0) = 0;
    Kokkos::deep_copy(KokkosExecSpace(),d_moduleStart,h_moduleStart);
    

    Kokkos::deep_copy(KokkosExecSpace(),d_id,h_id);
    Kokkos::deep_copy(KokkosExecSpace(),d_x,h_x);
    Kokkos::deep_copy(KokkosExecSpace(),d_y,h_y);
    Kokkos::deep_copy(KokkosExecSpace(),d_adc,h_adc);

    // Launch Kokkos Kernels
    std::cout << "Kokkos countModules kernel launch for " << n << " iterations\n";
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, n), KOKKOS_LAMBDA(const size_t i) {
          KOKKOS_NAMESPACE::gpuClustering::countModules(d_id, d_moduleStart, d_clus, n, i);
        });
    KokkosExecSpace().fence();

#ifdef KOKKOS_BACKEND_SERIAL
    uint32_t threadsPerModule = 1;
#else
    uint32_t threadsPerModule = (kkk == 5) ? 512 : ((kkk == 3) ? 128 : 256);
#endif
    uint32_t blocksPerGrid = MaxNumModules;  //nModules;

    std::cout << "Kokkos findModules kernel launch with " << blocksPerGrid << " blocks of " << threadsPerModule
              << " threads\n";
    
    Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(0,MaxNumModules),
      KOKKOS_LAMBDA(const size_t i){
        d_clusInModule(i) = 0;
      });

    gpuClustering::findClus(Kokkos::View<const uint16_t*, KokkosExecSpace>(d_id),
                            Kokkos::View<const uint16_t*, KokkosExecSpace>(d_x),
                            Kokkos::View<const uint16_t*, KokkosExecSpace>(d_y),
                            Kokkos::View<const uint32_t*, KokkosExecSpace>(d_moduleStart),
                            d_clusInModule,
                            d_moduleId,
                            d_clus,
                            n,
                            MaxNumModules,
                            threadsPerModule,
                            KokkosExecSpace());


    KokkosExecSpace().fence();
    Kokkos::deep_copy(KokkosExecSpace(),h_moduleStart,d_moduleStart);
    Kokkos::deep_copy(KokkosExecSpace(),h_clusInModule,d_clusInModule);

    auto clustInModule_acc = std::accumulate(&h_clusInModule(0), &h_clusInModule(0) + MaxNumModules, 0);
    std::cout << "before charge cut found " << clustInModule_acc << " clusters" << std::endl;

    for (auto i = MaxNumModules; i > 0; i--){
      if (h_clusInModule[i - 1] > 0) {
        std::cout << "last module is " << i - 1 << ' ' << h_clusInModule[i - 1] << std::endl;
        break;
      }
    }

    if (ncl != clustInModule_acc)
      std::cout << "ERROR!!!!! wrong number of cluster found" << std::endl;

    gpuClustering::clusterChargeCut(d_id,
                                    Kokkos::View<const uint16_t*, KokkosExecSpace>(d_adc),
                                    Kokkos::View<const uint32_t*, KokkosExecSpace>(d_moduleStart),
                                    d_clusInModule,
                                    Kokkos::View<const uint32_t*, KokkosExecSpace>(d_moduleId),
                                    d_clus,
                                    n,
                                    blocksPerGrid,
                                    threadsPerModule,
                                    KokkosExecSpace());
    KokkosExecSpace().fence();

    std::cout << "found " << h_moduleStart(0) << " Modules active" << std::endl;

    Kokkos::deep_copy(KokkosExecSpace(),h_id,d_id);
    Kokkos::deep_copy(KokkosExecSpace(),h_clus,d_clus);
    Kokkos::deep_copy(KokkosExecSpace(),h_clusInModule,d_clusInModule);
    Kokkos::deep_copy(KokkosExecSpace(),h_moduleId,d_moduleId);

    std::set<unsigned int> clids;
    for (int i = 0; i < n; ++i) {
      assert(h_id(i) != 666);  // only noise
      if (h_id(i) == InvId)
        continue;
      assert(h_clus(i) >= 0);
      clids.insert(h_id(i) * 1000 + h_clus(i));
    }

    // verify no hole in numbering
    auto p = clids.begin();
    auto cmid = (*p) / 1000;
    assert(0 == (*p) % 1000);
    auto c = p;
    ++c;
    std::cout << "first clusters " << *p << ' ' << *c << ' ' << h_clusInModule(cmid) << ' ' << h_clusInModule((*c) / 1000) << std::endl;
    std::cout << "last cluster " << *clids.rbegin() << ' ' << h_clusInModule((*clids.rbegin()) / 1000) << std::endl;
    for (; c != clids.end(); ++c) {
      auto cc = *c;
      auto pp = *p;
      auto mid = cc / 1000;
      auto pnc = pp % 1000;
      auto nc = cc % 1000;
      if (mid != cmid) {
        assert(0 == cc % 1000);
        assert(h_clusInModule(cmid) - 1 == pp % 1000);
        // if (h_clusInModule[cmid]-1 != pp%1000) std::cout << "error size " << mid << ": "  << h_clusInModule[mid] << ' ' << pp << std::endl;
        cmid = mid;
        p = c;
        continue;
      }
      p = c;
      // assert(nc==pnc+1);
      if (nc != pnc + 1)
        std::cout << "error " << mid << ": " << nc << ' ' << pnc << std::endl;
    }

    std::cout << "found " << std::accumulate(&h_clusInModule(0), &h_clusInModule(0) + MaxNumModules, 0) << ' ' << clids.size() << " clusters"
              << std::endl;
    for (auto i = MaxNumModules; i > 0; i--)
      if (h_clusInModule(i - 1) > 0) {
        std::cout << "last module is " << i - 1 << ' ' << h_clusInModule(i - 1) << std::endl;
        break;
      }
      // << " and " << seeds.size() << " seeds" << std::endl;

  }     /// end loop kkk
}

int main(void) {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize,1024*1024*1024);
  test();
  return 0;
}
