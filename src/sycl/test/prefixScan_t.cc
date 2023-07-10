#include <cassert>
#include <ios>
#include <iostream>

#include <sycl/sycl.hpp>

#include "SYCLCore/chooseDevice.h"
#include "SYCLCore/prefixScan.h"
#include "SYCLCore/printf.h"

template <typename T>
void testPrefixScan(sycl::nd_item<1> item, uint32_t size) {
  auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T[32]>(item.get_group());
  T *ws = (T *)wsbuff.get();
  auto cbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T[1024]>(item.get_group());
  T *c = (T *)cbuff.get();
  auto cobuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T[1024]>(item.get_group());
  T *co = (T *)cobuff.get();

  uint32_t first = item.get_local_id(0);
  for (auto i = first; i < size; i += item.get_local_range().get(0))
    c[i] = 1;
  sycl::group_barrier(item.get_group());

  cms::sycltools::blockPrefixScan(c, co, size, item, ws);
  cms::sycltools::blockPrefixScan(c, size, item, ws);

  //assert(1 == c[0]);
  if (1 != c[0]) {
    printf("failed (testPrefixScan): 1 != c[0]\n");
    return;
  }
  //assert(1 == co[0]);
  if (1 != co[0]) {
    printf("failed (testPrefixScan): 1 != co[0]\n");
    return;
  }

  for (uint32_t i = first + 1; i < size; i += item.get_local_range().get(0)) {
    if (c[i] != c[i - 1] + (T)1)
      printf("failed size %d, i %d, thread %lu, c[i] %d c[i - 1] %d", size, i, item.get_local_range(0), c[i], c[i - 1]);

    //assert(c[i] == c[i - 1] + 1);
    if (c[i] != c[i - 1] + (T)1) {
      printf("failed (testPrefixScan): c[i] != c[i - 1] + 1\n");
      return;
    }
    //assert(c[i] == i + 1);
    if (c[i] != i + (T)1) {
      printf("failed (testPrefixScan): c[i] != i + 1\n");
      return;
    }
    //assert(c[i] == co[i]);
    if (c[i] != co[i]) {
      printf("failed (testPrefixScan): c[i] != co[i]\n");
      return;
    }
  }
}

template <typename T>
void testWarpPrefixScan(sycl::nd_item<1> item, uint32_t size) {
  assert(size <= 32);

  auto cbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T[1024]>(item.get_group());
  T *c = (T *)cbuff.get();
  auto cobuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T[1024]>(item.get_group());
  T *co = (T *)cobuff.get();

  uint32_t i = item.get_local_id(0);
  c[i] = 1;
  sycl::group_barrier(item.get_group());

  warpPrefixScan(c, co, i, item);
  warpPrefixScan(c, i, item);
  sycl::group_barrier(item.get_group());

  //assert(1 == c[0]);
  if (1 != c[0]) {
    printf("failed (testWarpPrefixScan): 1 != c[0]\n");
    return;
  }
  //assert(1 == co[0]);
  if (1 != co[0]) {
    printf("failed (testWarpPrefixScan): 1 != co[0]\n");
    return;
  }

  if (i != 0) {
    if (c[i] != c[i - 1] + (T)1)
      printf("failed size %d, i %d, thread %lu, c[i] %d c[i - 1] %d", size, i, item.get_local_range(0), c[i], c[i - 1]);

    //assert(c[i] == c[i - 1] + 1);
    if (c[i] != c[i - 1] + (T)1) {
      printf("failed (testWarpPrefixScan): c[i] != c[i - 1] + 1\n");
      return;
    }
    //assert(c[i] == i + 1);
    if (c[i] != (T)i + (T)1) {
      printf("failed (testWarpPrefixScan): c[i] != i + 1\n");
      return;
    }
    //assert(c[i] == co[i]);
    if (c[i] != co[i]) {
      printf("failed (testWarpPrefixScan): c[i] != co[i]\n");
      return;
    }
  }
}

void init(sycl::nd_item<1> item, uint32_t *v, uint32_t val, uint32_t n) {
  auto i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (i < n)
    v[i] = val;
  if (i == 0)
    printf("init\n");
}

void verify(sycl::nd_item<1> item, uint32_t const *v, uint32_t n) {
  auto i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  if (i < n) {
    // assert(v[i] == i + 1);
    if (v[i] != i + 1) {
      printf("failed (verify): v[i] != i + 1 \n");
      return;
    }
  }
  if (i == 0)
    printf("verify\n");
}

int main(int argc, char **argv) try {
  std::string devices(argv[1]);
  setenv("SYCL_DEVICE_FILTER", devices.c_str(), true);

  cms::sycltools::enumerateDevices(true);
  sycl::device device = cms::sycltools::chooseDevice(0);
  sycl::queue queue = sycl::queue(device, sycl::property::queue::in_order());

  std::cout << "Prefixscan offload to " << device.get_info<sycl::info::device::name>() << " on backend "
            << device.get_backend() << std::endl;

  // query the device for the maximum workgroup size
  // FIXME the OpenCL CPU device reports a maximum workgroup size of 8192,
  // but workgroups bigger than 4096 result in a CL_OUT_OF_RESOURCES error
  const unsigned int maxWorkgroupSize = device.get_info<sycl::info::device::max_work_group_size>();
  std::cout << "max workgroup size: " << maxWorkgroupSize << std::endl;

  std::cout << "warp level" << std::endl;
  // std::cout << "warp 32" << std::endl;
  queue.submit([&](sycl::handler &cgh) {
    // sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
    // sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

    cgh.parallel_for<class testWarpPrefixScan_kernel_32>(
        sycl::nd_range<1>(sycl::range(32), sycl::range(32)),
        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { testWarpPrefixScan<int>(item, 32); });
  });
  queue.wait_and_throw();

  // std::cout << "warp 16" << std::endl;
  queue.submit([&](sycl::handler &cgh) {
    // sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
    // sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

    cgh.parallel_for<class testWarpPrefixScan_kernel_16>(
        sycl::nd_range<1>(sycl::range(32), sycl::range(32)),
        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { testWarpPrefixScan<int>(item, 16); });
  });
  queue.wait_and_throw();

  // std::cout << "warp 5" << std::endl;
  queue.submit([&](sycl::handler &cgh) {
    // sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
    // sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

    cgh.parallel_for<class testWarpPrefixScan_kernel_5>(
        sycl::nd_range<1>(sycl::range(32), sycl::range(32)),
        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { testWarpPrefixScan<int>(item, 5); });
  });
  queue.wait_and_throw();

  std::cout << "block level" << std::endl;
  for (unsigned int bs = 32; bs <= 256; bs += 32) {
    for (unsigned int j = 1; j <= 256; ++j) {
      queue.submit([&](sycl::handler &cgh) {
        // sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::target::local> ws_acc(32, cgh);
        // sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
        // sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

        cgh.parallel_for<class testPrefixScan_kernel_int>(
            sycl::nd_range<1>(sycl::range(bs), sycl::range(bs)),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { testPrefixScan<uint16_t>(item, j); });
      });
      queue.wait_and_throw();

      queue.submit([&](sycl::handler &cgh) {
        // sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::target::local> ws_acc(32, cgh);
        // sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
        // sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

        cgh.parallel_for<class testPrefixScan_kernel_float>(
            sycl::nd_range<1>(sycl::range(bs), sycl::range(bs)),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { testPrefixScan<float>(item, j); });
      });
      queue.wait_and_throw();
    }
  }

  // empiric limit
  auto max_items = maxWorkgroupSize * maxWorkgroupSize;
  unsigned int num_items = 10;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    std::cout << "multiblock" << std::endl;
    num_items *= 10;
    if (num_items > max_items) {
      std::cout << "Error: too many work items requested: " << num_items << " vs " << max_items << std::endl;
      break;
    }

    // declare, allocate, and initialize device-accessible pointers for input and output
    uint32_t *d_in = sycl::malloc_device<uint32_t>(num_items, queue);
    uint32_t *d_out = sycl::malloc_device<uint32_t>(num_items, queue);

    auto nthreads = 256;
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class init_kernel>(sycl::nd_range<1>(sycl::range(nblocks * nthreads), sycl::range(nthreads)),
                                          [=](sycl::nd_item<1> item) { init(item, d_in, 1, num_items); });
    });
    queue.wait_and_throw();

    // the block counter
    int32_t *d_pc = sycl::malloc_device<int32_t>(1, queue);

    nthreads = 1024;
    nblocks = (num_items + nthreads - 1) / nthreads;
    std::cout << "  nthreads: " << nthreads << " nblocks " << nblocks << " numitems " << num_items << std::endl;

    queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<unsigned int, 1, sycl::access_mode::read_write, sycl::target::local> psum_acc(4 * nblocks, cgh);

      cgh.parallel_for<class multiBlockPrefixScan_kernel>(
          sycl::nd_range(sycl::range(nblocks * nthreads), sycl::range(nthreads)),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
            cms::sycltools::multiBlockPrefixScan<uint32_t>(d_in, d_out, num_items, d_pc, item, psum_acc.get_pointer());
          });
    });
    queue.wait_and_throw();

    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class verify_kernel>(sycl::nd_range<1>(sycl::range(nblocks * nthreads), sycl::range(nthreads)),
                                            [=](sycl::nd_item<1> item) { verify(item, d_out, num_items); });
    });
    queue.wait_and_throw();
  }  // ksize

  return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
