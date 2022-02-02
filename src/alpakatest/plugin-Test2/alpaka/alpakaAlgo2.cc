#include "alpakaAlgo2.h"
#include "AlpakaCore/alpakaDevices.h"
#include "AlpakaCore/alpakaWorkDiv.h"

namespace {
  constexpr unsigned int NUM_VALUES = 1000;

  struct vectorAdd {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc,
                                  const T_Data* __restrict__ a,
                                  const T_Data* __restrict__ b,
                                  T_Data* __restrict__ c,
                                  unsigned int numElements) const {
      // Global element index in 1D grid.
      // NB: On GPU, i = threadIndexGlobal = firstElementIdxGlobal = endElementIdxGlobal.
      cms::alpakatools::for_each_element_in_grid(acc, numElements, [&](uint32_t i) { c[i] = a[i] + b[i]; });
    }
  };

  struct vectorProd {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc,
                                  const T_Data* __restrict__ a,
                                  const T_Data* __restrict__ b,
                                  T_Data* __restrict__ c,
                                  unsigned int numElements) const {
      // Global element index in 2D grid.
      // NB: On GPU, threadIndexGlobal = firstElementIdxGlobal = endElementIdxGlobal.
      const auto [firstElementIdxGlobalCol, endElementIdxGlobalCol] =
          cms::alpakatools::element_index_range_in_grid_truncated(acc, numElements, 0, 0);
      const auto [firstElementIdxGlobalRow, endElementIdxGlobalRow] =
          cms::alpakatools::element_index_range_in_grid_truncated(acc, numElements, 0, 1);

      for (uint32_t col = firstElementIdxGlobalCol; col < endElementIdxGlobalCol; ++col) {
        for (uint32_t row = firstElementIdxGlobalRow; row < endElementIdxGlobalRow; ++row) {
          c[row + numElements * col] = a[row] * b[col];
        }
      }
    }
  };

  struct matrixMul {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc,
                                  const T_Data* __restrict__ a,
                                  const T_Data* __restrict__ b,
                                  T_Data* __restrict__ c,
                                  unsigned int numElements) const {
      // Global element index in 2D grid.
      // NB: On GPU, threadIndexGlobal = firstElementIdxGlobal = endElementIdxGlobal.
      const auto [firstElementIdxGlobalCol, endElementIdxGlobalCol] =
          cms::alpakatools::element_index_range_in_grid_truncated(acc, numElements, 0, 0);
      const auto [firstElementIdxGlobalRow, endElementIdxGlobalRow] =
          cms::alpakatools::element_index_range_in_grid_truncated(acc, numElements, 0, 1);

      for (uint32_t col = firstElementIdxGlobalCol; col < endElementIdxGlobalCol; ++col) {
        for (uint32_t row = firstElementIdxGlobalRow; row < endElementIdxGlobalRow; ++row) {
          T_Data tmp = 0;
          for (unsigned int i = 0; i < numElements; ++i) {
            tmp += a[row + numElements * i] * b[i + numElements * col];
          }
          c[row + numElements * col] = tmp;
        }
      }
    }
  };

  struct matrixMulVector {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc,
                                  const T_Data* __restrict__ a,
                                  const T_Data* __restrict__ b,
                                  T_Data* __restrict__ c,
                                  unsigned int numElements) const {
      // Global element index in 1D grid.
      // NB: On GPU, threadIndexGlobal = firstElementIdxGlobal = endElementIdxGlobal.
      cms::alpakatools::for_each_element_in_grid(acc, numElements, [&](uint32_t row) {
        T_Data tmp = 0;
        for (unsigned int i = 0; i < numElements; ++i) {
          tmp += a[row * numElements + i] * b[i];
        }
        c[row] = tmp;
      });
    }
  };

  /* 
     DEBUG ONLY
     Obviously not optimized (and contains printf anyway), incorporated to verify results.
  */
  namespace debug {
    constexpr float TOLERANCE_RATIO = 0.01;

    struct verifyVectorAdd {
      template <typename T_Acc, typename T_Data>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc, const T_Data* result, unsigned int numElements) const {
        // Global element index in 1D grid.
        // NB: On GPU, i = threadIndexGlobal = firstElementIdxGlobal = endElementIdxGlobal.
        cms::alpakatools::for_each_element_in_grid(acc, numElements, [&](uint32_t i) {
          // theoreticalResult = i+i^2 = i*(i+1)
          if (result[i] != i * (i + 1)) {
            printf("Wrong vectorAdd results, i = %u, c[i] = %f.\n", i, result[i]);
          }
        });
      }
    };

    struct verifyVectorProd {
      template <typename T_Acc, typename T_Data>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc, const T_Data* result, unsigned int numElements) const {
        const auto& threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc));
        const uint32_t threadIdxGlobalX(threadIdxGlobal[0u]);
        const uint32_t threadIdxGlobalY(threadIdxGlobal[1u]);

        if (threadIdxGlobalX == 0 && threadIdxGlobalY == 0) {
          for (unsigned int row = 0; row < numElements; ++row) {
            for (unsigned int col = 0; col < numElements; ++col) {
              const T_Data theoreticalResult = static_cast<T_Data>(row) * col * col;
              const T_Data diff = result[row + numElements * col] - theoreticalResult;
              const T_Data tolerance = theoreticalResult * TOLERANCE_RATIO;
              if (diff > tolerance || diff < -tolerance) {
                printf("Wrong vectorProd results, row = %u, col = %u, c[row,col] = %f.\n",
                       row,
                       col,
                       result[row * numElements + col]);
              }
            }
          }
        }
      }
    };

    struct verifyMatrixMul {
      template <typename T_Acc, typename T_Data>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc, const T_Data* result, unsigned int numElements) const {
        const auto& threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc));
        const uint32_t threadIdxGlobalX(threadIdxGlobal[0u]);
        const uint32_t threadIdxGlobalY(threadIdxGlobal[1u]);

        if (threadIdxGlobalX == 0 && threadIdxGlobalY == 0) {
          const T_Data partialtheoreticalResult =
              static_cast<T_Data>(numElements - 1) * (numElements - 1) * numElements * numElements / 4;

          for (unsigned int row = 0; row < numElements; ++row) {
            for (unsigned int col = 0; col < numElements; ++col) {
              // theoreticalResult = row * col * (col+1) * Sum(k=1 to numElements-1, k^3)
              const T_Data theoreticalResult = static_cast<T_Data>(row) * col * (col + 1) * partialtheoreticalResult;
              const T_Data diff = result[row + numElements * col] - theoreticalResult;
              const T_Data tolerance = theoreticalResult * TOLERANCE_RATIO;

              if (diff > tolerance || diff < -tolerance) {
                printf(
                    "Wrong matrix multiplication results, row = %u, col = %u, c[row,col] = %f, theoreticalResult = "
                    "%f.\n",
                    row,
                    col,
                    result[row * numElements + col],
                    theoreticalResult);
              }
            }
          }
        }
      }
    };

    struct verifyMatrixMulVector {
      template <typename T_Acc, typename T_Data>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc, const T_Data* result, unsigned int numElements) const {
        const uint32_t threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

        if (threadIdxGlobal == 0) {
          const unsigned long int N = numElements - 1;
          const unsigned long int N2 = N * N;
          const unsigned long int N3 = N2 * N;
          const unsigned long int N4 = N2 * N2;
          const unsigned long int N5 = N4 * N;
          const T_Data partialtheoreticalResult =
              static_cast<T_Data>(N2) * (N + 1) * (N + 1) / 4 *
              ((6 * N5 + 15 * N4 + 10 * N3 - N) / 30 + (N + 1) * (N + 1) * N2 / 2 + N * (N + 1) / 2);

          for (unsigned int i = 0; i < numElements; ++i) {
            // theoreticalResult = N^2*(N+1)^2/4 * i * Sum(k=1 to N, k^2*(k+1)^2)
            const T_Data theoreticalResult = i * partialtheoreticalResult;
            const T_Data diff = result[i] - theoreticalResult;
            const T_Data tolerance = theoreticalResult * TOLERANCE_RATIO;

            if (diff > tolerance || diff < -tolerance) {
              printf("Wrong matrix-vector multiplication results, i = %u, c[i] = %f, theoreticalResult = %f.\n",
                     i,
                     result[i],
                     theoreticalResult);
            }
          }
        }
      }
    };

  }  // namespace debug

}  // namespace

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  cms::alpakatools::device_buffer<Device, float[]> alpakaAlgo2() {
    const auto& device = cms::alpakatools::devices<Platform>[0];
    Queue queue(device);

    // Host data
    auto h_a_buf = cms::alpakatools::make_host_buffer<float[]>(NUM_VALUES);
    auto h_b_buf = cms::alpakatools::make_host_buffer<float[]>(NUM_VALUES);
    auto h_a = h_a_buf.data();
    auto h_b = h_b_buf.data();
    for (auto i = 0U; i < NUM_VALUES; i++) {
      h_a[i] = i;
      h_b[i] = i * i;
    }

    // Device data
    auto d_a_buf = cms::alpakatools::make_device_buffer<float[]>(queue, NUM_VALUES);
    auto d_b_buf = cms::alpakatools::make_device_buffer<float[]>(queue, NUM_VALUES);
    alpaka::memcpy(queue, d_a_buf, h_a_buf);
    alpaka::memcpy(queue, d_b_buf, h_b_buf);
    auto d_c_buf = cms::alpakatools::make_device_buffer<float[]>(queue, NUM_VALUES);

    // Prepare 1D workDiv
    const Vec1D& blocksPerGrid1(Vec1D::all((NUM_VALUES + 32 - 1) / 32));
    const Vec1D& threadsPerBlockOrElementsPerThread1(Vec1D(32u));
    const WorkDiv1D& workDiv1 =
        cms::alpakatools::make_workdiv<Acc1D>(blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

    // VECTOR ADDITION
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(
                        workDiv1, vectorAdd(), d_a_buf.data(), d_b_buf.data(), d_c_buf.data(), NUM_VALUES));

    // Prepare 2D workDiv
    const unsigned int blocksPerGridSide = (NUM_VALUES <= 32 ? 1 : std::ceil(NUM_VALUES / 32.));
    const Vec2D& blocksPerGrid2(Vec2D::all(blocksPerGridSide));
    const unsigned int threadsPerBlockOrElementsPerThreadSide = (NUM_VALUES < 32 ? NUM_VALUES : 32u);
    const Vec2D& threadsPerBlockOrElementsPerThread2(Vec2D::all(threadsPerBlockOrElementsPerThreadSide));
    const WorkDiv2D& workDiv2 =
        cms::alpakatools::make_workdiv<Acc2D>(blocksPerGrid2, threadsPerBlockOrElementsPerThread2);

    // Device data
    auto d_ma_buf = cms::alpakatools::make_device_buffer<float[]>(queue, NUM_VALUES * NUM_VALUES);
    auto d_mb_buf = cms::alpakatools::make_device_buffer<float[]>(queue, NUM_VALUES * NUM_VALUES);
    auto d_mc_buf = cms::alpakatools::make_device_buffer<float[]>(queue, NUM_VALUES * NUM_VALUES);

    // VECTOR MULTIPLICATION
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc2D>(
                        workDiv2, vectorProd(), d_a_buf.data(), d_b_buf.data(), d_ma_buf.data(), NUM_VALUES));

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc2D>(
                        workDiv2, vectorProd(), d_a_buf.data(), d_c_buf.data(), d_mb_buf.data(), NUM_VALUES));

    // MATRIX MULTIPLICATION
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc2D>(
                        workDiv2, matrixMul(), d_ma_buf.data(), d_mb_buf.data(), d_mc_buf.data(), NUM_VALUES));

    // MATRIX - VECTOR MULTIPLICATION
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(
                        workDiv1, matrixMulVector(), d_mc_buf.data(), d_b_buf.data(), d_c_buf.data(), NUM_VALUES));

    alpaka::wait(queue);
    return d_mc_buf;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
