#ifndef ALPAKAWORKDIVHELPER_H
#define ALPAKAWORKDIVHELPER_H

#include "AlpakaCore/alpakaConfig.h"

using namespace alpaka_common;

namespace cms {
  namespace alpakatools {

    /*
     * Creates the accelerator-dependent workdiv.
     */
    template <typename T_Dim>
    WorkDiv<T_Dim> make_workdiv(const Vec<T_Dim>& blocksPerGrid, const Vec<T_Dim>& threadsPerBlockOrElementsPerThread) {
      // On the GPU:
      // threadsPerBlockOrElementsPerThread is the number of threads per block.
      // Each thread is looking at a single element: elementsPerThread is always 1.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      const Vec<T_Dim>& elementsPerThread = Vec<T_Dim>::ones();
      return WorkDiv<T_Dim>(blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
#else
      // On the CPU:
      // Run serially with a single thread per block: threadsPerBlock is always 1.
      // threadsPerBlockOrElementsPerThread is the number of elements per thread.
      const Vec<T_Dim>& threadsPerBlock = Vec<T_Dim>::ones();
      return WorkDiv<T_Dim>(blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
#endif
    }

    /*
     * Computes the range of the element(s) global index(es) in grid.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_global_index_range(const T_Acc& acc) {
      Vec<T_Dim> firstElementIdxGlobalVec = Vec<T_Dim>::zeros();
      Vec<T_Dim> endElementIdxUncutGlobalVec = Vec<T_Dim>::zeros();

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        // Global thread index in grid (along dimension dimIndex).
        const uint32_t threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[dimIndex]);
        const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[dimIndex]);

        // Global element index in grid (along dimension dimIndex).
        // Obviously relevant for CPU only.
        // For GPU, threadDimension = 1, and firstElementIdxGlobal = endElementIdxGlobal = threadIndexGlobal.
        const uint32_t firstElementIdxGlobal = threadIdxGlobal * threadDimension;
        const uint32_t endElementIdxUncutGlobal = firstElementIdxGlobal + threadDimension;

        firstElementIdxGlobalVec[dimIndex] = firstElementIdxGlobal;
        endElementIdxUncutGlobalVec[dimIndex] = endElementIdxUncutGlobal;
      }

      return {firstElementIdxGlobalVec, endElementIdxUncutGlobalVec};
    }

    /*
     * Computes the range of the element(s) global index(es) in grid.
     * Truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_global_index_range_truncated(const T_Acc& acc,
											   const Vec<T_Dim>& maxNumberOfElements) {
      static_assert(alpaka::dim::Dim<T_Acc>::value == T_Dim::value,
                    "Accelerator and maxNumberOfElements need to have same dimension.");
      auto&& [firstElementIdxGlobalVec, endElementIdxGlobalVec] = element_global_index_range(acc);

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        endElementIdxGlobalVec[dimIndex] = std::min(endElementIdxGlobalVec[dimIndex], maxNumberOfElements[dimIndex]);
      }

      return {firstElementIdxGlobalVec, endElementIdxGlobalVec};
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAWORKDIVHELPER_H
