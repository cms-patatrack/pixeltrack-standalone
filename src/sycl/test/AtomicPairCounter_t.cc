#include <cassert>

#include <CL/sycl.hpp>

#include "SYCLCore/AtomicPairCounter.h"
#include "SYCLCore/chooseDevice.h"

using AtomicPairCounter = cms::sycltools::AtomicPairCounter;

void update(AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n, sycl::nd_item<1> item) {
  auto i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  if (i >= n)
    return;

  auto m = i % 11;
  m = m % 6 + 1;  // max 6, no 0
  auto c = dc->add(m);
  //assert(c.m < n);
  ind[c.m] = c.n;
  for (auto j = c.n; j < c.n + m; ++j)
    cont[j] = i;
};

void finalize(AtomicPairCounter const *dc, uint32_t *ind, uint32_t *cont, uint32_t n) {
  //assert(dc->get().m == n);
  ind[n] = dc->get().n;
}

void verify(AtomicPairCounter const *dc, uint32_t const *ind, uint32_t const *cont, uint32_t n, sycl::nd_item<1> item) {
  auto i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  if (i >= n)
    return;
  //assert(0 == ind[0]);
  //assert(dc->get().m == n);
  //assert(ind[n] == dc->get().n);
  // auto ib = ind[i];
  // auto ie = ind[i + 1];
  // auto k = cont[ib++];
  //assert(k < n);
  //for (; ib < ie; ++ib)
  //assert(cont[ib] == k);
}

#include <iostream>
int main(int argc, char **argv) {
  std::string devices(argv[1]);
  setenv("SYCL_DEVICE_FILTER", devices.c_str(), true);

  cms::sycltools::enumerateDevices(true);
  sycl::device device = cms::sycltools::chooseDevice(0);
  sycl::queue queue = sycl::queue(device, sycl::property::queue::in_order());

  std::cout << "AtomicPairCounter offload to " << device.get_info<cl::sycl::info::device::name>() << " on backend "
            << device.get_backend() << std::endl;

  AtomicPairCounter *dc_d = sycl::malloc_device<AtomicPairCounter>(1, queue);
  queue.memset(dc_d, 0, sizeof(AtomicPairCounter)).wait();

  std::cout << "size " << sizeof(AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  uint32_t *n_d, *m_d;
  n_d = (uint32_t *)sycl::malloc_device(N * sizeof(int), queue);
  m_d = (uint32_t *)sycl::malloc_device(M * sizeof(int), queue);

  int max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  int threads = std::min(512, max_work_group_size);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(2000 * threads, threads),
                     [=](sycl::nd_item<1> item) { update(dc_d, n_d, m_d, 10000, item); });
  });
  queue.submit([&](sycl::handler &cgh) { cgh.single_task([=]() { finalize(dc_d, n_d, m_d, 10000); }); });
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(2000 * threads, threads),
                     [=](sycl::nd_item<1> item) { verify(dc_d, n_d, m_d, 10000, item); });
  });

  AtomicPairCounter dc;
  queue.memcpy(&dc, dc_d, sizeof(AtomicPairCounter)).wait();

  std::cout << dc.get().n << ' ' << dc.get().m << std::endl;

  return 0;
}
