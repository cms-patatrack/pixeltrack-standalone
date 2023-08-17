#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include <array>
#include <memory>

#include <sycl/sycl.hpp>

#include "SYCLCore/AtomicPairCounter.h"
#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/chooseDevice.h"
#include "SYCLCore/device_unique_ptr.h"

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;

using Assoc = cms::sycltools::OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;
using SmallAssoc = cms::sycltools::OneToManyAssoc<uint16_t, 128, MaxAssocs>;
using Multiplicity = cms::sycltools::OneToManyAssoc<uint16_t, 8, MaxTk>;
using TK = std::array<uint16_t, 4>;
using cms::sycltools::AtomicPairCounter;

void countMultiLocal(TK const* __restrict__ tk, Multiplicity* __restrict__ assoc, int32_t n, sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int i = first; i < n; i += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto localbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<Multiplicity::CountersOnly>(item.get_group());
    Multiplicity::CountersOnly* local = (Multiplicity::CountersOnly*)localbuff.get();
    if (item.get_local_id(0) == 0)
      local->zero();
    sycl::group_barrier(item.get_group());
    local->countDirect(2 + i % 4);
    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) == 0)
      assoc->add(*local);
  }
}

void countMulti(TK const* __restrict__ tk, Multiplicity* __restrict__ assoc, int32_t n, sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int i = first; i < n; i += item.get_group_range(0) * item.get_local_range().get(0))
    assoc->countDirect(2 + i % 4);
}

void verifyMulti(Multiplicity* __restrict__ m1, Multiplicity* __restrict__ m2, sycl::nd_item<1> item) {
  auto first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (auto i = first; i < Multiplicity::totbins(); i += item.get_group_range(0) * item.get_local_range().get(0))
    assert(m1->off[i] == m2->off[i]);
}

void count(TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n, sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int i = first; i < 4 * n; i += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto k = i / 4;
    auto j = i - 4 * k;
    assert(j < 4);
    if (k >= n)
      return;
    if (tk[k][j] < MaxElem)
      assoc->countDirect(tk[k][j]);
  }
}

void fill(TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n, sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int i = first; i < 4 * n; i += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto k = i / 4;
    auto j = i - 4 * k;
    assert(j < 4);
    if (k >= n)
      return;
    if (tk[k][j] < MaxElem)
      assoc->fillDirect(tk[k][j], k);
  }
}

void verify(Assoc* __restrict__ assoc) { assert(assoc->size() < Assoc::capacity()); }

template <typename Assoc>
void fillBulk(
    AtomicPairCounter* apc, TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n, sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int k = first; k < n; k += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto m = tk[k][3] < MaxElem ? 4 : 3;
    assoc->bulkFill(*apc, &tk[k][0], m);
  }
}

template <typename Assoc>
void verifyBulk(Assoc const* __restrict__ assoc, AtomicPairCounter const* apc) {
  if (apc->get().m >= Assoc::nbins())
    printf("Overflow %d %d\n", apc->get().m, Assoc::nbins());
  assert(assoc->size() < Assoc::capacity());
}

int main(int argc, char** argv) {
  std::cout << "OneToManyAssoc " << sizeof(Assoc) << ' ' << Assoc::nbins() << ' ' << Assoc::capacity() << std::endl;
  std::cout << "OneToManyAssoc (small) " << sizeof(SmallAssoc) << ' ' << SmallAssoc::nbins() << ' '
            << SmallAssoc::capacity() << std::endl;

  std::mt19937 eng;
  std::geometric_distribution<int> rdm(0.8);

  constexpr uint32_t N = 4000;
  std::vector<std::array<uint16_t, 4>> tr(N);

  // fill with "index" to element
  long long ave = 0;
  int imax = 0;
  auto n = 0U;
  auto z = 0U;
  auto nz = 0U;
  for (auto i = 0U; i < 4U; ++i) {
    auto j = 0U;
    while (j < N && n < MaxElem) {
      if (z == 11) {
        ++n;
        z = 0;
        ++nz;
        continue;
      }  // a bit of not assoc
      auto x = rdm(eng);
      auto k = std::min(j + x + 1, N);
      if (i == 3 && z == 3) {  // some triplets time to time
        for (; j < k; ++j)
          tr[j][i] = MaxElem + 1;
      } else {
        ave += x + 1;
        imax = std::max(imax, x);
        for (; j < k; ++j)
          tr[j][i] = n;
        ++n;
      }
      ++z;
    }
    assert(n <= MaxElem);
    assert(j <= N);
  }
  std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;

  std::string devices(argv[1]);
  setenv("ONEAPI_DEVICE_SELECTOR", devices.c_str(), true);

  cms::sycltools::enumerateDevices(true);
  sycl::device device = cms::sycltools::chooseDevice(0);
  sycl::queue queue = sycl::queue(device, sycl::property::queue::in_order());
  std::cout << "OneToMany offload to " << device.get_info<sycl::info::device::name>() << " on backend "
            << device.get_backend() << std::endl;

  auto v_d = cms::sycltools::make_device_unique<std::array<uint16_t, 4>[]>(N, queue);
  assert(v_d.get());
  auto a_d = cms::sycltools::make_device_unique<Assoc[]>(1, queue);
  auto sa_d = cms::sycltools::make_device_unique<SmallAssoc[]>(1, queue);

  queue.memcpy(v_d.get(), tr.data(), N * sizeof(std::array<uint16_t, 4>)).wait();

  cms::sycltools::launchZero(a_d.get(), queue);

  int max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  auto nThreads = std::min(256, max_work_group_size);
  auto nBlocks = (4 * N + nThreads - 1) / nThreads;

  queue.submit([&](sycl::handler& cgh) {
    auto v_d_get = v_d.get();
    auto a_d_get = a_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(nBlocks * nThreads), sycl::range(nThreads)),
                     [=](sycl::nd_item<1> item) { count(v_d_get, a_d_get, N, item); });
  });

  cms::sycltools::launchFinalize(a_d.get(), queue);
  queue.submit([&](sycl::handler& cgh) {
    auto a_d_get = a_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(1), sycl::range(1)), [=](sycl::nd_item<1> item) { verify(a_d_get); });
  });
  queue.submit([&](sycl::handler& cgh) {
    auto v_d_get = v_d.get();
    auto a_d_get = a_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(nBlocks * nThreads), sycl::range(nThreads)),
                     [=](sycl::nd_item<1> item) { fill(v_d_get, a_d_get, N, item); });
  });

  Assoc la;

  queue.memcpy(&la, a_d.get(), sizeof(Assoc)).wait();

  std::cout << la.size() << std::endl;
  imax = 0;
  ave = 0;
  z = 0;
  for (auto i = 0U; i < n; ++i) {
    auto x = la.size(i);
    if (x == 0) {
      z++;
      continue;
    }
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la.size(n));
  std::cout << "found with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << z << std::endl;

  // now the inverse map (actually this is the direct....)
  AtomicPairCounter* dc_d;
  AtomicPairCounter dc(0);

  dc_d = sycl::malloc_device<AtomicPairCounter>(1, queue);
  queue.memset(dc_d, 0, sizeof(AtomicPairCounter)).wait();
  nBlocks = (N + nThreads - 1) / nThreads;
  queue.submit([&](sycl::handler& cgh) {
    auto v_d_get = v_d.get();
    auto a_d_get = a_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(nBlocks * nThreads), sycl::range(nThreads)),
                     [=](sycl::nd_item<1> item) { fillBulk(dc_d, v_d_get, a_d_get, N, item); });
  });
  queue.submit([&](sycl::handler& cgh) {
    auto a_d_get = a_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(nBlocks * nThreads), sycl::range(nThreads)),
                     [=](sycl::nd_item<1> item) { cms::sycltools::finalizeBulk(dc_d, a_d_get, item); });
  });
  queue.submit([&](sycl::handler& cgh) {
    auto a_d_get = a_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(1), sycl::range(1)),
                     [=](sycl::nd_item<1> item) { verifyBulk(a_d_get, dc_d); });
  });

  queue.memcpy(&la, a_d.get(), sizeof(Assoc));
  queue.memcpy(&dc, dc_d, sizeof(AtomicPairCounter));

  queue.memset(dc_d, 0, sizeof(AtomicPairCounter)).wait();
  queue.submit([&](sycl::handler& cgh) {
    auto v_d_get = v_d.get();
    auto sa_d_get = sa_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(nBlocks * nThreads), sycl::range(nThreads)),
                     [=](sycl::nd_item<1> item) { fillBulk(dc_d, v_d_get, sa_d_get, N, item); });
  });
  queue.submit([&](sycl::handler& cgh) {
    auto sa_d_get = sa_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(nBlocks * nThreads), sycl::range(nThreads)),
                     [=](sycl::nd_item<1> item) { cms::sycltools::finalizeBulk(dc_d, sa_d_get, item); });
  });
  queue.submit([&](sycl::handler& cgh) {
    sycl::stream out(64 * 1024, 80, cgh);

    auto sa_d_get = sa_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(1), sycl::range(1)),
                     [=](sycl::nd_item<1> item) { verifyBulk(sa_d_get, dc_d); });
  });

  std::cout << "final counter value " << dc.get().n << ' ' << dc.get().m << std::endl;

  std::cout << la.size() << std::endl;
  imax = 0;
  ave = 0;
  for (auto i = 0U; i < N; ++i) {
    auto x = la.size(i);
    if (!(x == 4 || x == 3))
      std::cout << i << ' ' << x << std::endl;
    assert(x == 4 || x == 3);
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la.size(N));
  std::cout << "found with ave occupancy " << double(ave) / N << ' ' << imax << std::endl;

  // here verify use of block local counters
  auto m1_d = cms::sycltools::make_device_unique<Multiplicity[]>(1, queue);
  auto m2_d = cms::sycltools::make_device_unique<Multiplicity[]>(1, queue);
  cms::sycltools::launchZero(m1_d.get(), queue);
  cms::sycltools::launchZero(m2_d.get(), queue);

  nBlocks = (4 * N + nThreads - 1) / nThreads;
  queue.submit([&](sycl::handler& cgh) {
    auto v_d_get = v_d.get();
    auto m1_d_get = m1_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(nBlocks * nThreads), sycl::range(nThreads)),
                     [=](sycl::nd_item<1> item) { countMulti(v_d_get, m1_d_get, N, item); });
  });
  queue.submit([&](sycl::handler& cgh) {
    auto v_d_get = v_d.get();
    auto m2_d_get = m2_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(nBlocks * nThreads), sycl::range(nThreads)),
                     [=](sycl::nd_item<1> item) { countMultiLocal(v_d_get, m2_d_get, N, item); });
  });
  queue.submit([&](sycl::handler& cgh) {
    auto m1_d_get = m1_d.get();
    auto m2_d_get = m2_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(Multiplicity::totbins()), sycl::range(Multiplicity::totbins())),
                     [=](sycl::nd_item<1> item) { verifyMulti(m1_d_get, m2_d_get, item); });
  });

  cms::sycltools::launchFinalize(m1_d.get(), queue);
  cms::sycltools::launchFinalize(m2_d.get(), queue);
  queue.submit([&](sycl::handler& cgh) {
    auto m1_d_get = m1_d.get();
    auto m2_d_get = m2_d.get();

    cgh.parallel_for(sycl::nd_range(sycl::range(Multiplicity::totbins()), sycl::range(Multiplicity::totbins())),
                     [=](sycl::nd_item<1> item) { verifyMulti(m1_d_get, m2_d_get, item); });
  });

  queue.wait_and_throw();

  return 0;
}
