#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include <CL/sycl.hpp>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/chooseDevice.h"
#include "SYCLCore/printf.h"

using namespace cms::sycltools;

template <typename T, int NBINS, int S, int DELTA>
void mykernel(T const* __restrict__ v, uint32_t N, sycl::nd_item<1> item) {
  assert(v);
  assert(N == 12000);

  if (item.get_local_id(0) == 0)
    printf("start kernel for %d data\n", N);

  using Hist = HistoContainer<T, NBINS, 12000, S, uint16_t>;
  using counter = typename Hist::Counter;

  auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<counter[32]>(item.get_group());
  counter* ws = (counter*)wsbuff.get();
  auto histbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<Hist>(item.get_group());
  Hist* hist = (Hist*)histbuff.get();

  for (auto j = item.get_local_id(0); j < Hist::totbins(); j += item.get_local_range().get(0)) {
    hist->off[j] = 0;
  }
  sycl::group_barrier(item.get_group());

  for (auto j = item.get_local_id(0); j < N; j += item.get_local_range().get(0))
    hist->count(v[j]);
  sycl::group_barrier(item.get_group());

  assert(0 == hist->size());
  sycl::group_barrier(item.get_group());

  hist->finalize(item, ws);
  sycl::group_barrier(item.get_group());

  assert(N == hist->size());
  for (auto j = item.get_local_id(0); j < Hist::nbins(); j += item.get_local_range().get(0))
    assert(hist->off[j] <= hist->off[j + 1]);
  sycl::group_barrier(item.get_group());

  if (item.get_local_id(0) < 32)
    ws[item.get_local_id(0)] = 0;  // used by prefix scan...
  sycl::group_barrier(item.get_group());

  for (auto j = item.get_local_id(0); j < N; j += item.get_local_range().get(0))
    hist->fill(v[j], j);
  sycl::group_barrier(item.get_group());
  assert(0 == hist->off[0]);
  assert(N == hist->size());

  for (auto j = item.get_local_id(0); j < hist->size() - 1; j += item.get_local_range().get(0)) {
    auto p = hist->begin() + j;
    assert((*p) < N);
    [[maybe_unused]] auto k1 = Hist::bin(v[*p]);
    [[maybe_unused]] auto k2 = Hist::bin(v[*(p + 1)]);
    assert(k2 >= k1);
  }

  for (auto i = item.get_local_id(0); i < hist->size(); i += item.get_local_range().get(0)) {
    auto p = hist->begin() + i;
    auto j = *p;
    auto b0 = Hist::bin(v[j]);
    [[maybe_unused]] int tot = 0;
    auto ftest = [&](int k) {
      assert(k >= 0 && k < (int)N);
      ++tot;
    };
    forEachInWindow(*hist, v[j], v[j], ftest);
    [[maybe_unused]] int rtot = hist->size(b0);
    assert(tot == rtot);
    tot = 0;
    auto vm = int(v[j]) - DELTA;
    auto vp = int(v[j]) + DELTA;
    constexpr int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
    vm = sycl::max(vm, 0);
    vm = sycl::min(vm, vmax);
    vp = sycl::min(vp, vmax);
    vp = sycl::max(vp, 0);
    assert(vp >= vm);
    forEachInWindow(*hist, vm, vp, ftest);
    int bp = Hist::bin(vp);
    int bm = Hist::bin(vm);
    rtot = hist->end(bp) - hist->begin(bm);
    assert(tot == rtot);
  }
}

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go(sycl::queue queue) {
  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::mt19937 eng;
  std::uniform_int_distribution<T> rgen(rmin, rmax);

  constexpr int N = 12000;
  T v[N];

  auto v_d = cms::sycltools::make_device_unique<T[]>(N, queue);
  assert(v_d.get());

  using Hist = cms::sycltools::HistoContainer<T, NBINS, N, S, uint16_t>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::capacity() << ' '
            << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;

    assert(v_d.get());
    assert(v);
    queue.memcpy(v_d.get(), v, N * sizeof(T)).wait();

    int max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    int nthreads = std::min(256, max_work_group_size);
    queue.submit([&](sycl::handler& cgh) {
      auto v_d_get = v_d.get();

      cgh.parallel_for(sycl::nd_range<1>(sycl::range(nthreads), sycl::range(nthreads)),
                       [=](sycl::nd_item<1> item) { mykernel<T, NBINS, S, DELTA>(v_d_get, N, item); });
    });

    queue.wait_and_throw();
  }
}

int main(int argc, char** argv) {
  std::string devices(argv[1]);
  setenv("SYCL_DEVICE_FILTER", devices.c_str(), true);

  cms::sycltools::enumerateDevices(true);
  sycl::device device = cms::sycltools::chooseDevice(0);
  sycl::queue queue = sycl::queue(device, sycl::property::queue::in_order());

  std::cout << "HistoContainer offload to " << device.get_info<cl::sycl::info::device::name>() << " on backend "
            << device.get_backend() << std::endl;

  std::cout << "test <int16_t>" << std::endl;
  go<int16_t>(queue);

  std::cout << "test <uint8_t, 128, 8, 4>" << std::endl;
  go<uint8_t, 128, 8, 4>(queue);

  std::cout << "test <uint16_t, 313 / 2, 9, 4>" << std::endl;
  go<uint16_t, 313 / 2, 9, 4>(queue);

  std::cout << "done" << std::endl;
  return 0;
}
