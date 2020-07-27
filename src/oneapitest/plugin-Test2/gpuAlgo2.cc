#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gpuAlgo2.h"

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include <cmath>

namespace {
  constexpr int NUM_VALUES = 1000;

  template <typename T>
  void vectorAdd(const T *a, const T *b, T *c, int numElements, ::sycl::nd_item<3> item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    if (i < numElements) {
      c[i] = a[i] + b[i];
    }
  }

  template <typename T>
  void vectorProd(const T *a, const T *b, T *c, int numElements, ::sycl::nd_item<3> item_ct1) {
    int row = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1);
    int col = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

    if (row < numElements && col < numElements) {
      c[row * numElements + col] = a[row] * b[col];
    }
  }

  template <typename T>
  void matrixMul(const T *a, const T *b, T *c, int numElements, ::sycl::nd_item<3> item_ct1) {
    int row = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1);
    int col = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

    if (row < numElements && col < numElements) {
      T tmp = 0;
      for (int i = 0; i < numElements; ++i) {
        tmp += a[row * numElements + i] * b[i * numElements + col];
      }
      c[row * numElements + col] = tmp;
    }
  }

  template <typename T>
  void matrixMulVector(const T *a, const T *b, T *c, int numElements, ::sycl::nd_item<3> item_ct1) {
    int row = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1);

    if (row < numElements) {
      T tmp = 0;
      for (int i = 0; i < numElements; ++i) {
        tmp += a[row * numElements + i] * b[i];
      }
      c[row] = tmp;
    }
  }
}  // namespace

cms::sycl::device::unique_ptr<float[]> gpuAlgo2(::sycl::queue stream) {
  auto h_a = cms::sycl::make_host_unique<float[]>(NUM_VALUES, stream);
  auto h_b = cms::sycl::make_host_unique<float[]>(NUM_VALUES, stream);

  for (auto i = 0; i < NUM_VALUES; i++) {
    h_a[i] = i;
    h_b[i] = i * i;
  }

  auto d_a = cms::sycl::make_device_unique<float[]>(NUM_VALUES, stream);
  auto d_b = cms::sycl::make_device_unique<float[]>(NUM_VALUES, stream);

  stream.memcpy(d_a.get(), h_a.get(), NUM_VALUES * sizeof(float));
  stream.memcpy(d_b.get(), h_b.get(), NUM_VALUES * sizeof(float));

  int threadsPerBlock{32};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  auto d_c = cms::sycl::make_device_unique<float[]>(NUM_VALUES, stream);
  stream.submit([&](::sycl::handler &cgh) {
    auto d_a_get_ct0 = d_a.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_c_get_ct2 = d_c.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(
        ::sycl::nd_range<3>(::sycl::range<3>(1, 1, blocksPerGrid) * ::sycl::range<3>(1, 1, threadsPerBlock),
                          ::sycl::range<3>(1, 1, threadsPerBlock)),
        [=](::sycl::nd_item<3> item_ct1) { vectorAdd(d_a_get_ct0, d_b_get_ct1, d_c_get_ct2, NUM_VALUES_ct3, item_ct1); });
  });

  auto d_ma = cms::sycl::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mb = cms::sycl::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mc = cms::sycl::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  ::sycl::range<3> threadsPerBlock3{NUM_VALUES, NUM_VALUES, 1};
  ::sycl::range<3> blocksPerGrid3{1, 1, 1};
  if (NUM_VALUES * NUM_VALUES > 32) {
    threadsPerBlock3[0] = 32;
    threadsPerBlock3[1] = 32;
    blocksPerGrid3[0] = ceil(double(NUM_VALUES) / double(threadsPerBlock3[0]));
    blocksPerGrid3[1] = ceil(double(NUM_VALUES) / double(threadsPerBlock3[1]));
  }
  stream.submit([&](::sycl::handler &cgh) {
    auto dpct_global_range = blocksPerGrid3 * threadsPerBlock3;

    auto d_a_get_ct0 = d_a.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_ma_get_ct2 = d_ma.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(
        ::sycl::nd_range<3>(::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                          ::sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](::sycl::nd_item<3> item_ct1) {
          vectorProd(d_a_get_ct0, d_b_get_ct1, d_ma_get_ct2, NUM_VALUES_ct3, item_ct1);
        });
  });
  stream.submit([&](::sycl::handler &cgh) {
    auto dpct_global_range = blocksPerGrid3 * threadsPerBlock3;

    auto d_a_get_ct0 = d_a.get();
    auto d_c_get_ct1 = d_c.get();
    auto d_mb_get_ct2 = d_mb.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(
        ::sycl::nd_range<3>(::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                          ::sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](::sycl::nd_item<3> item_ct1) {
          vectorProd(d_a_get_ct0, d_c_get_ct1, d_mb_get_ct2, NUM_VALUES_ct3, item_ct1);
        });
  });
  stream.submit([&](::sycl::handler &cgh) {
    auto dpct_global_range = blocksPerGrid3 * threadsPerBlock3;

    auto d_ma_get_ct0 = d_ma.get();
    auto d_mb_get_ct1 = d_mb.get();
    auto d_mc_get_ct2 = d_mc.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(
        ::sycl::nd_range<3>(::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                          ::sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](::sycl::nd_item<3> item_ct1) {
          matrixMul(d_ma_get_ct0, d_mb_get_ct1, d_mc_get_ct2, NUM_VALUES_ct3, item_ct1);
        });
  });

  stream.submit([&](::sycl::handler &cgh) {
    auto d_mc_get_ct0 = d_mc.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_c_get_ct2 = d_c.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(::sycl::nd_range<3>(::sycl::range<3>(1, 1, blocksPerGrid) * ::sycl::range<3>(1, 1, threadsPerBlock),
                                       ::sycl::range<3>(1, 1, threadsPerBlock)),
                     [=](::sycl::nd_item<3> item_ct1) {
                       matrixMulVector(d_mc_get_ct0, d_b_get_ct1, d_c_get_ct2, NUM_VALUES_ct3, item_ct1);
                     });
  });

  return d_a;
}
