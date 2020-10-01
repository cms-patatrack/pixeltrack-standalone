#include <algorithm>
#include <cmath>

#include <CL/sycl.hpp>

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "gpuAlgo2.h"

namespace gpu_algo_2 {
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

using namespace gpu_algo_2;

constexpr unsigned int sqrt2(unsigned int value) {
  unsigned int result = 1;
  while (value >= 4) {
    value /= 4;
    result *= 2;
  }
  return result;
}

cms::sycltools::device::unique_ptr<float[]> gpuAlgo2(sycl::queue stream) {
  // query the device for the maximum "block size" or workgroup size
  const unsigned int workgroupSize = stream.get_device().get_info<sycl::info::device::max_work_group_size>();

  // build a rectangular "block" or workgroup
  const unsigned int workgroupRectB = sqrt2(workgroupSize);
  const unsigned int workgroupRectA = workgroupSize / workgroupRectB;

  auto h_a = cms::sycltools::make_host_unique<float[]>(NUM_VALUES, stream);
  auto h_b = cms::sycltools::make_host_unique<float[]>(NUM_VALUES, stream);

  for (auto i = 0; i < NUM_VALUES; i++) {
    h_a[i] = i;
    h_b[i] = i * i;
  }

  auto d_a = cms::sycltools::make_device_unique<float[]>(NUM_VALUES, stream);
  auto d_b = cms::sycltools::make_device_unique<float[]>(NUM_VALUES, stream);

  stream.memcpy(d_a.get(), h_a.get(), NUM_VALUES * sizeof(float));
  stream.memcpy(d_b.get(), h_b.get(), NUM_VALUES * sizeof(float));

  unsigned int threadsPerBlock = std::min<unsigned int>(NUM_VALUES, workgroupSize);
  unsigned int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  std::cerr << "block size: " << threadsPerBlock << " / " << stream.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl; 
  std::cerr << "grid size:  " << blocksPerGrid   << std::endl;

  auto d_c = cms::sycltools::make_device_unique<float[]>(NUM_VALUES, stream);
  stream.submit([&](sycl::handler &cgh) {
    auto d_a_get_ct0 = d_a.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_c_get_ct2 = d_c.get();

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                                       sycl::range<3>(1, 1, threadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) {
                       vectorAdd(d_a_get_ct0, d_b_get_ct1, d_c_get_ct2, NUM_VALUES, item_ct1);
                     });
  });

  auto d_ma = cms::sycltools::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mb = cms::sycltools::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mc = cms::sycltools::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);

  sycl::range<3> threadsPerBlock3{workgroupRectA, workgroupRectB, 1};
  sycl::range<3> blocksPerGrid3{(NUM_VALUES + threadsPerBlock3[0] - 1) / threadsPerBlock3[0], (NUM_VALUES + threadsPerBlock3[1] - 1) / threadsPerBlock3[1], 1};
  stream.submit([&](sycl::handler &cgh) {
    auto threadsPerGrid = blocksPerGrid3 * threadsPerBlock3;

    auto d_a_get_ct0 = d_a.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_ma_get_ct2 = d_ma.get();

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(threadsPerGrid.get(2), threadsPerGrid.get(1), threadsPerGrid.get(0)),
                          sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          vectorProd(d_a_get_ct0, d_b_get_ct1, d_ma_get_ct2, NUM_VALUES, item_ct1);
        });
  });
  
  stream.submit([&](sycl::handler &cgh) {
    auto threadsPerGrid = blocksPerGrid3 * threadsPerBlock3;

    auto d_a_get_ct0 = d_a.get();
    auto d_c_get_ct1 = d_c.get();
    auto d_mb_get_ct2 = d_mb.get();

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(threadsPerGrid.get(2), threadsPerGrid.get(1), threadsPerGrid.get(0)),
                          sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          vectorProd(d_a_get_ct0, d_c_get_ct1, d_mb_get_ct2, NUM_VALUES, item_ct1);
        });
  });
  
  stream.submit([&](sycl::handler &cgh) {
    auto threadsPerGrid = blocksPerGrid3 * threadsPerBlock3;

    auto d_ma_get_ct0 = d_ma.get();
    auto d_mb_get_ct1 = d_mb.get();
    auto d_mc_get_ct2 = d_mc.get();

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(threadsPerGrid.get(2), threadsPerGrid.get(1), threadsPerGrid.get(0)),
                          sycl::range<3>(threadsPerBlock3.get(2), threadsPerBlock3.get(1), threadsPerBlock3.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          matrixMul(d_ma_get_ct0, d_mb_get_ct1, d_mc_get_ct2, NUM_VALUES, item_ct1);
        });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto d_mc_get_ct0 = d_mc.get();
    auto d_b_get_ct1 = d_b.get();
    auto d_c_get_ct2 = d_c.get();

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                                       sycl::range<3>(1, 1, threadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) {
                       matrixMulVector(d_mc_get_ct0, d_b_get_ct1, d_c_get_ct2, NUM_VALUES, item_ct1);
                     });
  });

  return d_a;
}
