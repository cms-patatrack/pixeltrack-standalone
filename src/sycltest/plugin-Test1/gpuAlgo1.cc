#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gpuAlgo1.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include <cmath>

namespace {
  constexpr int NUM_VALUES = 1000;

  template <typename T>
  SYCL_EXTERNAL void vectorAdd(const T *a, const T *b, T *c, int numElements, sycl::nd_item<3> item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    if (i < numElements) {
      c[i] = a[i] + b[i];
    }
  }

  template <typename T>
  SYCL_EXTERNAL void vectorProd(const T *a, const T *b, T *c, int numElements, sycl::nd_item<3> item_ct1) {
    int row = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1);
    int col = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

    if (row < numElements && col < numElements) {
      c[row * numElements + col] = a[row] * b[col];
    }
  }

  template <typename T>
  SYCL_EXTERNAL void matrixMul(const T *a, const T *b, T *c, int numElements, sycl::nd_item<3> item_ct1) {
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
  SYCL_EXTERNAL void matrixMulVector(const T *a, const T *b, T *c, int numElements, sycl::nd_item<3> item_ct1) {
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

cms::cuda::device::unique_ptr<float[]> gpuAlgo1(sycl::queue *stream) {
  auto h_a = cms::cuda::make_host_unique<float[]>(NUM_VALUES, stream);
  auto h_b = cms::cuda::make_host_unique<float[]>(NUM_VALUES, stream);

  for (auto i = 0; i < NUM_VALUES; i++) {
    h_a[i] = i;
    h_b[i] = i * i;
  }

  auto d_a = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);
  auto d_b = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);

  /*
  DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(d_a.get(), h_a.get(), NUM_VALUES * sizeof(float)), 0));
  /*
  DPCT1003:38: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(d_b.get(), h_b.get(), NUM_VALUES * sizeof(float)), 0));

  int threadsPerBlock{32};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  auto d_c = cms::cuda::make_device_unique<float[]>(NUM_VALUES, stream);
  auto current_device = cms::cuda::currentDevice();
  /*
  DPCT1049:39: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    auto d_a_get_ct0 = d_a.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_c_get_ct2 = d_c.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                                       sycl::range<3>(1, 1, threadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) {
                       vectorAdd(d_a_get_ct0, d_b_get_ct1, d_c_get_ct2, NUM_VALUES_ct3, item_ct1);
                     });
  });

  auto d_ma = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mb = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mc = cms::cuda::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  sycl::range<3> threadsPerBlock3{NUM_VALUES, NUM_VALUES, 1};
  sycl::range<3> blocksPerGrid3{1, 1, 1};
  if (NUM_VALUES * NUM_VALUES > 32) {
    threadsPerBlock3[0] = 32;
    threadsPerBlock3[1] = 32;
    blocksPerGrid3[0] = ceil(double(NUM_VALUES) / double(threadsPerBlock3[0]));
    blocksPerGrid3[1] = ceil(double(NUM_VALUES) / double(threadsPerBlock3[1]));
  }
  /*
  DPCT1049:34: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    auto dpct_global_range = blocksPerGrid3 * threadsPerBlock3;

    auto d_a_get_ct0 = d_a.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_ma_get_ct2 = d_ma.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                          sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          vectorProd(d_a_get_ct0, d_b_get_ct1, d_ma_get_ct2, NUM_VALUES_ct3, item_ct1);
        });
  });
  /*
  DPCT1049:35: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    auto dpct_global_range = blocksPerGrid3 * threadsPerBlock3;

    auto d_a_get_ct0 = d_a.get();
    auto d_c_get_ct1 = d_c.get();
    auto d_mb_get_ct2 = d_mb.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                          sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          vectorProd(d_a_get_ct0, d_c_get_ct1, d_mb_get_ct2, NUM_VALUES_ct3, item_ct1);
        });
  });
  /*
  DPCT1049:36: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    auto dpct_global_range = blocksPerGrid3 * threadsPerBlock3;

    auto d_ma_get_ct0 = d_ma.get();
    auto d_mb_get_ct1 = d_mb.get();
    auto d_mc_get_ct2 = d_mc.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                          sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          matrixMul(d_ma_get_ct0, d_mb_get_ct1, d_mc_get_ct2, NUM_VALUES_ct3, item_ct1);
        });
  });

  /*
  DPCT1049:40: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    auto d_mc_get_ct0 = d_mc.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_c_get_ct2 = d_c.get();
    auto NUM_VALUES_ct3 = NUM_VALUES;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                                       sycl::range<3>(1, 1, threadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) {
                       matrixMulVector(d_mc_get_ct0, d_b_get_ct1, d_c_get_ct2, NUM_VALUES_ct3, item_ct1);
                     });
  });

  return d_a;
}
