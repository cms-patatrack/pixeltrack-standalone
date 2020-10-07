#include <algorithm>
#include <cmath>

#include <CL/sycl.hpp>

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "gpuAlgo2.h"

namespace gpu_algo_2 {
  constexpr int NUM_VALUES = 1000;

  template <typename T>
  SYCL_EXTERNAL void vectorAdd(const T *a, const T *b, T *c, int numElements, sycl::nd_item<1> item) {
    int i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
    if (i < numElements) {
      c[i] = a[i] + b[i];
    }
  }

  template <typename T>
  SYCL_EXTERNAL void vectorProd(const T *a, const T *b, T *c, int numElements, sycl::nd_item<3> item) {
    int row = item.get_group(1) * item.get_local_range().get(1) + item.get_local_id(1);
    int col = item.get_group(2) * item.get_local_range().get(2) + item.get_local_id(2);

    if (row < numElements && col < numElements) {
      c[row * numElements + col] = a[row] * b[col];
    }
  }

  template <typename T>
  SYCL_EXTERNAL void matrixMul(const T *a, const T *b, T *c, int numElements, sycl::nd_item<3> item) {
    int row = item.get_group(1) * item.get_local_range().get(1) + item.get_local_id(1);
    int col = item.get_group(2) * item.get_local_range().get(2) + item.get_local_id(2);

    if (row < numElements && col < numElements) {
      T tmp = 0;
      for (int i = 0; i < numElements; ++i) {
        tmp += a[row * numElements + i] * b[i * numElements + col];
      }
      c[row * numElements + col] = tmp;
    }
  }

  template <typename T>
  SYCL_EXTERNAL void matrixMulVector(const T *a, const T *b, T *c, int numElements, sycl::nd_item<1> item) {
    int row = item.get_group(1) * item.get_local_range().get(1) + item.get_local_id(1);

    if (row < numElements) {
      T tmp = 0;
      for (int i = 0; i < numElements; ++i) {
        tmp += a[row * numElements + i] * b[i];
      }
      c[row] = tmp;
    }
  }
}  // namespace gpu_algo_2

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
  unsigned int threadsPerGrid = threadsPerGrid;

  std::cerr << "block size: " << threadsPerBlock << " / "
            << stream.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
  std::cerr << "grid size:  " << blocksPerGrid << std::endl;

  auto d_c = cms::sycltools::make_device_unique<float[]>(NUM_VALUES, stream);
  stream.submit([&](sycl::handler &cgh) {
    auto pd_a = d_a.get();
    auto pd_b = d_b.get();
    auto pd_c = d_c.get();

    cgh.parallel_for(sycl::nd_range<1>(threadsPerGrid, threadsPerBlock),
                     [=](sycl::nd_item<1> item) { vectorAdd(pd_a, pd_b, pd_c, NUM_VALUES, item); });
  });

  auto d_ma = cms::sycltools::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mb = cms::sycltools::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);
  auto d_mc = cms::sycltools::make_device_unique<float[]>(NUM_VALUES * NUM_VALUES, stream);

  sycl::range<3> threadsPerBlock3{1, workgroupRectB, workgroupRectA};
  sycl::range<3> blocksPerGrid3{
      1, (NUM_VALUES + workgroupRectB - 1) / workgroupRectB, (NUM_VALUES + workgroupRectA - 1) / workgroupRectA};
  auto threadsPerGrid3 = blocksPerGrid3 * threadsPerBlock3;

  stream.submit([&](sycl::handler &cgh) {
    auto pd_a = d_a.get();
    auto pd_b = d_b.get();
    auto pd_ma = d_ma.get();

    cgh.parallel_for(sycl::nd_range<3>(threadsPerGrid3, threadsPerBlock3),
                     [=](sycl::nd_item<3> item) { vectorProd(pd_a, pd_b, pd_ma, NUM_VALUES, item); });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto pd_a = d_a.get();
    auto pd_c = d_c.get();
    auto pd_mb = d_mb.get();

    cgh.parallel_for(sycl::nd_range<3>(threadsPerGrid3, threadsPerBlock3),
                     [=](sycl::nd_item<3> item) { vectorProd(pd_a, pd_c, pd_mb, NUM_VALUES, item); });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto pd_ma = d_ma.get();
    auto pd_mb = d_mb.get();
    auto pd_mc = d_mc.get();

    cgh.parallel_for(sycl::nd_range<3>(threadsPerGrid3, threadsPerBlock3),
                     [=](sycl::nd_item<3> item) { matrixMul(pd_ma, pd_mb, pd_mc, NUM_VALUES, item); });
  });

  stream.submit([&](sycl::handler &cgh) {
    auto pd_mc = d_mc.get();
    auto pd_b = d_b.get();
    auto pd_c = d_c.get();

    cgh.parallel_for(sycl::nd_range<1>(threadsPerGrid, threadsPerBlock),
                     [=](sycl::nd_item<1> item) { matrixMulVector(pd_mc, pd_b, pd_c, NUM_VALUES, item); });
  });

  return d_a;
}
