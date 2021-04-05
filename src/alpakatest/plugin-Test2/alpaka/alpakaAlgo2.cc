#include "alpakaAlgo2.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"

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
      cms::alpakatools::for_each_element_in_thread_1D_index_in_grid(
          acc, numElements, [&](uint32_t i) { c[i] = a[i] + b[i]; });
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
      const auto& [firstElementIdxGlobal, endElementIdxGlobal] =
          cms::alpakatools::element_index_range_in_grid_truncated(acc, Vec2::all(numElements));

      for (uint32_t col = firstElementIdxGlobal[0u]; col < endElementIdxGlobal[0u]; ++col) {
        for (uint32_t row = firstElementIdxGlobal[1u]; row < endElementIdxGlobal[1u]; ++row) {
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
      const auto& [firstElementIdxGlobal, endElementIdxGlobal] =
          cms::alpakatools::element_index_range_in_grid_truncated(acc, Vec2::all(numElements));

      for (uint32_t col = firstElementIdxGlobal[0u]; col < endElementIdxGlobal[0u]; ++col) {
        for (uint32_t row = firstElementIdxGlobal[1u]; row < endElementIdxGlobal[1u]; ++row) {
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
      cms::alpakatools::for_each_element_in_thread_1D_index_in_grid(acc, numElements, [&](uint32_t row) {
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
        cms::alpakatools::for_each_element_in_thread_1D_index_in_grid(acc, numElements, [&](uint32_t i) {
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
  AlpakaAccBuf2<float> alpakaAlgo2() {
    const DevHost host(alpaka::getDevByIdx<PltfHost>(0u));
    const DevAcc2 device(alpaka::getDevByIdx<PltfAcc2>(0u));
    const Vec1 size(NUM_VALUES);
    Queue queue(device);

    // Host data
    auto h_a_buf = alpaka::allocBuf<float, Idx>(host, size);
    auto h_b_buf = alpaka::allocBuf<float, Idx>(host, size);
    auto h_a = alpaka::getPtrNative(h_a_buf);
    auto h_b = alpaka::getPtrNative(h_b_buf);
    for (auto i = 0U; i < NUM_VALUES; i++) {
      h_a[i] = i;
      h_b[i] = i * i;
    }

    // Device data
    auto d_a_buf = alpaka::allocBuf<float, Idx>(device, size);
    auto d_b_buf = alpaka::allocBuf<float, Idx>(device, size);
    alpaka::memcpy(queue, d_a_buf, h_a_buf, size);
    alpaka::memcpy(queue, d_b_buf, h_b_buf, size);
    auto d_c_buf = alpaka::allocBuf<float, Idx>(device, size);

    // Prepare 1D workDiv
    const Vec1& blocksPerGrid1(Vec1::all((NUM_VALUES + 32 - 1) / 32));
    const Vec1& threadsPerBlockOrElementsPerThread1(Vec1(32u));
    const WorkDiv1& workDiv1 = cms::alpakatools::make_workdiv(blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

    // VECTOR ADDITION
    alpaka::enqueue(queue,
                           alpaka::createTaskKernel<Acc1>(workDiv1,
                                                                  vectorAdd(),
                                                                  alpaka::getPtrNative(d_a_buf),
                                                                  alpaka::getPtrNative(d_b_buf),
                                                                  alpaka::getPtrNative(d_c_buf),
                                                                  NUM_VALUES));

    // Prepare 2D workDiv
    const unsigned int blocksPerGridSide = (NUM_VALUES <= 32 ? 1 : std::ceil(NUM_VALUES / 32.));
    const Vec2& blocksPerGrid2(Vec2::all(blocksPerGridSide));
    const unsigned int threadsPerBlockOrElementsPerThreadSide = (NUM_VALUES < 32 ? NUM_VALUES : 32u);
    const Vec2& threadsPerBlockOrElementsPerThread2(Vec2::all(threadsPerBlockOrElementsPerThreadSide));
    const WorkDiv2& workDiv2 = cms::alpakatools::make_workdiv(blocksPerGrid2, threadsPerBlockOrElementsPerThread2);

    // Device data
    const Vec2 sizeSquare(NUM_VALUES, NUM_VALUES);
    auto d_ma_buf = alpaka::allocBuf<float, Idx>(device, sizeSquare);
    auto d_mb_buf = alpaka::allocBuf<float, Idx>(device, sizeSquare);
    auto d_mc_buf = alpaka::allocBuf<float, Idx>(device, sizeSquare);

    // VECTOR MULTIPLICATION
    alpaka::enqueue(queue,
                           alpaka::createTaskKernel<Acc2>(workDiv2,
                                                                  vectorProd(),
                                                                  alpaka::getPtrNative(d_a_buf),
                                                                  alpaka::getPtrNative(d_b_buf),
                                                                  alpaka::getPtrNative(d_ma_buf),
                                                                  NUM_VALUES));

    alpaka::enqueue(queue,
                           alpaka::createTaskKernel<Acc2>(workDiv2,
                                                                  vectorProd(),
                                                                  alpaka::getPtrNative(d_a_buf),
                                                                  alpaka::getPtrNative(d_c_buf),
                                                                  alpaka::getPtrNative(d_mb_buf),
                                                                  NUM_VALUES));

    // MATRIX MULTIPLICATION
    alpaka::enqueue(queue,
                           alpaka::createTaskKernel<Acc2>(workDiv2,
                                                                  matrixMul(),
                                                                  alpaka::getPtrNative(d_ma_buf),
                                                                  alpaka::getPtrNative(d_mb_buf),
                                                                  alpaka::getPtrNative(d_mc_buf),
                                                                  NUM_VALUES));

    // MATRIX - VECTOR MULTIPLICATION
    alpaka::enqueue(queue,
                           alpaka::createTaskKernel<Acc1>(workDiv1,
                                                                  matrixMulVector(),
                                                                  alpaka::getPtrNative(d_mc_buf),
                                                                  alpaka::getPtrNative(d_b_buf),
                                                                  alpaka::getPtrNative(d_c_buf),
                                                                  NUM_VALUES));

    alpaka::wait(queue);
    return d_mc_buf;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
