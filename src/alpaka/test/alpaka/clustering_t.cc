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

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"

// dirty, but works
#include "plugin-SiPixelClusterizer/alpaka/gpuClustering.h"
#include "plugin-SiPixelClusterizer/alpaka/gpuClusterChargeCut.h"

int main(void) {
  const DevHost host(alpaka::getDevByIdx<PltfHost>(0u));
  const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1 device(alpaka::getDevByIdx<ALPAKA_ACCELERATOR_NAMESPACE::PltfAcc1>(0u));
  ALPAKA_ACCELERATOR_NAMESPACE::Queue queue(device);

  constexpr unsigned int numElements = 256 * 2000;
  // these in reality are already on GPU
  auto h_id_buf = alpaka::allocBuf<uint16_t, Idx>(host, numElements);
  auto h_id = alpaka::getPtrNative(h_id_buf);
  auto h_x_buf = alpaka::allocBuf<uint16_t, Idx>(host, numElements);
  auto h_x = alpaka::getPtrNative(h_x_buf);
  auto h_y_buf = alpaka::allocBuf<uint16_t, Idx>(host, numElements);
  auto h_y = alpaka::getPtrNative(h_y_buf);
  auto h_adc_buf = alpaka::allocBuf<uint16_t, Idx>(host, numElements);
  auto h_adc = alpaka::getPtrNative(h_adc_buf);

  auto h_clus_buf = alpaka::allocBuf<int, Idx>(host, numElements);
  auto h_clus = alpaka::getPtrNative(h_clus_buf);

  auto d_id_buf = alpaka::allocBuf<uint16_t, Idx>(device, numElements);
  auto d_x_buf = alpaka::allocBuf<uint16_t, Idx>(device, numElements);
  auto d_y_buf = alpaka::allocBuf<uint16_t, Idx>(device, numElements);
  auto d_adc_buf = alpaka::allocBuf<uint16_t, Idx>(device, numElements);
  auto d_clus_buf = alpaka::allocBuf<int, Idx>(device, numElements);

  auto d_moduleStart_buf = alpaka::allocBuf<uint32_t, Idx>(device, gpuClustering::MaxNumModules + 1);
  auto d_clusInModule_buf = alpaka::allocBuf<uint32_t, Idx>(device, gpuClustering::MaxNumModules);
  auto d_moduleId_buf = alpaka::allocBuf<uint32_t, Idx>(device, gpuClustering::MaxNumModules);

  // later random number
  unsigned int n = 0;
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
      h_id[n++] = gpuClustering::InvId;  // error
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
        h_id[n++] = gpuClustering::InvId;  // error
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
              h_id[n++] = gpuClustering::InvId;
              h_id[n++] = gpuClustering::InvId;
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
  for (auto kkk = 0; kkk < 5; ++kkk) {
    n = 0;
    ncl = 0;
    generateClusters(kkk);

    std::cout << "created " << n << " digis in " << ncl << " clusters" << std::endl;
    assert(n <= numElements);

    auto h_nModules_buf = alpaka::allocBuf<uint32_t, Idx>(host, 1u);
    auto nModules = alpaka::getPtrNative(h_nModules_buf);
    nModules[0] = 0;
    alpaka::memcpy(queue, d_moduleStart_buf, h_nModules_buf, 1u);

    alpaka::memcpy(queue, d_id_buf, h_id_buf, n);
    alpaka::memcpy(queue, d_x_buf, h_x_buf, n);
    alpaka::memcpy(queue, d_y_buf, h_y_buf, n);
    alpaka::memcpy(queue, d_adc_buf, h_adc_buf, n);

// Launch CUDA Kernels
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    const int threadsPerBlockOrElementsPerThread = (kkk == 5) ? 512 : ((kkk == 3) ? 128 : 256);
#else
    // NB: can be tuned.
    const int threadsPerBlockOrElementsPerThread = 256;
#endif

    // COUNT MODULES
    const int blocksPerGridCountModules =
        (numElements + threadsPerBlockOrElementsPerThread - 1) / threadsPerBlockOrElementsPerThread;
    const WorkDiv1& workDivCountModules = cms::alpakatools::make_workdiv(Vec1::all(blocksPerGridCountModules),
                                                                         Vec1::all(threadsPerBlockOrElementsPerThread));
    std::cout << "CUDA countModules kernel launch with " << blocksPerGridCountModules << " blocks of "
              << threadsPerBlockOrElementsPerThread << " threads (GPU) or elements (CPU). \n";

    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDivCountModules,
                                                                     gpuClustering::countModules(),
                                                                     alpaka::getPtrNative(d_id_buf),
                                                                     alpaka::getPtrNative(d_moduleStart_buf),
                                                                     alpaka::getPtrNative(d_clus_buf),
                                                                     n));

    // FIND CLUSTER
    const WorkDiv1& workDivMaxNumModules = cms::alpakatools::make_workdiv(
        Vec1::all(gpuClustering::MaxNumModules), Vec1::all(threadsPerBlockOrElementsPerThread));
    std::cout << "CUDA findModules kernel launch with " << gpuClustering::MaxNumModules << " blocks of "
              << threadsPerBlockOrElementsPerThread << " threads (GPU) or elements (CPU). \n";

    alpaka::memset(queue, d_clusInModule_buf, 0, gpuClustering::MaxNumModules);

    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDivMaxNumModules,
                                                                     gpuClustering::findClus(),
                                                                     alpaka::getPtrNative(d_id_buf),
                                                                     alpaka::getPtrNative(d_x_buf),
                                                                     alpaka::getPtrNative(d_y_buf),
                                                                     alpaka::getPtrNative(d_moduleStart_buf),
                                                                     alpaka::getPtrNative(d_clusInModule_buf),
                                                                     alpaka::getPtrNative(d_moduleId_buf),
                                                                     alpaka::getPtrNative(d_clus_buf),
                                                                     n));
    alpaka::memcpy(queue, h_nModules_buf, d_moduleStart_buf, 1u);

    auto h_nclus_buf = alpaka::allocBuf<uint32_t, Idx>(host, gpuClustering::MaxNumModules);
    auto nclus = alpaka::getPtrNative(h_nclus_buf);
    alpaka::memcpy(queue, h_nclus_buf, d_clusInModule_buf, gpuClustering::MaxNumModules);

    // Wait for memory transfers to be completed
    alpaka::wait(queue);

    auto h_moduleId_buf = alpaka::allocBuf<uint32_t, Idx>(host, nModules[0]);
    //auto moduleId = alpaka::getPtrNative(h_moduleId_buf);

    std::cout << "before charge cut found " << std::accumulate(nclus, nclus + gpuClustering::MaxNumModules, 0)
              << " clusters" << std::endl;
    for (auto i = gpuClustering::MaxNumModules; i > 0; i--)
      if (nclus[i - 1] > 0) {
        std::cout << "last module is " << i - 1 << ' ' << nclus[i - 1] << std::endl;
        break;
      }
    if (ncl != std::accumulate(nclus, nclus + gpuClustering::MaxNumModules, 0))
      std::cout << "ERROR!!!!! wrong number of cluster found" << std::endl;

    // CLUSTER CHARGE CUT
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDivMaxNumModules,
                                                                     gpuClustering::clusterChargeCut(),
                                                                     alpaka::getPtrNative(d_id_buf),
                                                                     alpaka::getPtrNative(d_adc_buf),
                                                                     alpaka::getPtrNative(d_moduleStart_buf),
                                                                     alpaka::getPtrNative(d_clusInModule_buf),
                                                                     alpaka::getPtrNative(d_moduleId_buf),
                                                                     alpaka::getPtrNative(d_clus_buf),
                                                                     n));
    alpaka::memcpy(queue, h_id_buf, d_id_buf, n);
    alpaka::memcpy(queue, h_clus_buf, d_clus_buf, n);
    alpaka::memcpy(queue, h_nclus_buf, d_clusInModule_buf, gpuClustering::MaxNumModules);
    alpaka::memcpy(queue, h_moduleId_buf, d_moduleId_buf, nModules[0]);

    // Wait for memory transfers to be completed
    alpaka::wait(queue);
    std::cout << "found " << nModules[0] << " Modules active" << std::endl;

    // CROSS-CHECK
    std::set<unsigned int> clids;
    for (unsigned int i = 0; i < n; ++i) {
      assert(h_id[i] != 666);  // only noise
      if (h_id[i] == gpuClustering::InvId)
        continue;
      assert(h_clus[i] >= 0);
      assert(h_clus[i] < int(nclus[h_id[i]]));
      clids.insert(h_id[i] * 1000 + h_clus[i]);
      // clids.insert(h_clus[i]);
    }

    // verify no hole in numbering
    auto p = clids.begin();
    auto cmid = (*p) / 1000;
    assert(0 == (*p) % 1000);
    auto c = p;
    ++c;
    std::cout << "first clusters " << *p << ' ' << *c << ' ' << nclus[cmid] << ' ' << nclus[(*c) / 1000] << std::endl;
    std::cout << "last cluster " << *clids.rbegin() << ' ' << nclus[(*clids.rbegin()) / 1000] << std::endl;
    for (; c != clids.end(); ++c) {
      auto cc = *c;
      auto pp = *p;
      auto mid = cc / 1000;
      auto pnc = pp % 1000;
      auto nc = cc % 1000;
      if (mid != cmid) {
        assert(0 == cc % 1000);
        assert(nclus[cmid] - 1 == pp % 1000);
        // if (nclus[cmid]-1 != pp%1000) std::cout << "error size " << mid << ": "  << nclus[mid] << ' ' << pp << std::endl;
        cmid = mid;
        p = c;
        continue;
      }
      p = c;
      // assert(nc==pnc+1);
      if (nc != pnc + 1)
        std::cout << "error " << mid << ": " << nc << ' ' << pnc << std::endl;
    }

    std::cout << "found " << std::accumulate(nclus, nclus + gpuClustering::MaxNumModules, 0) << ' ' << clids.size()
              << " clusters" << std::endl;
    for (auto i = gpuClustering::MaxNumModules; i > 0; i--)
      if (nclus[i - 1] > 0) {
        std::cout << "last module is " << i - 1 << ' ' << nclus[i - 1] << std::endl;
        break;
      }
    // << " and " << seeds.size() << " seeds" << std::endl;
  }  /// end loop kkk
  return 0;
}
