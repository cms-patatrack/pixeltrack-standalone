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
     * 1D helper to only access 1 element per block 
     * (should obviously only be needed for debug / printout).
     */
    template <typename T_Acc>
    ALPAKA_FN_ACC bool once_per_block_1D(const T_Acc& acc, uint32_t i) {
      const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      return (i % blockDimension == 0);
    }

    /*
     * Computes the range of the elements indexes, local to the block.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim = alpaka::Dim<T_Acc>>
    ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_block(const T_Acc& acc,
                                                                                 const Vec<T_Dim>& elementIdxShift) {
      Vec<T_Dim> firstElementIdxVec = Vec<T_Dim>::zeros();
      Vec<T_Dim> endElementIdxUncutVec = Vec<T_Dim>::zeros();

      // Loop on all grid dimensions.
      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        // Take into account the thread index in block.
        const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndex]);
        const uint32_t threadDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[dimIndex]);

        // Compute the elements indexes in block.
        // Obviously relevant for CPU only.
        // For GPU, threadDimension = 1, and elementIdx = firstElementIdx = threadIdx + elementIdxShift.
        const uint32_t firstElementIdxLocal = threadIdxLocal * threadDimension;
        const uint32_t firstElementIdx = firstElementIdxLocal + elementIdxShift[dimIndex];  // Add the shift!
        const uint32_t endElementIdxUncut = firstElementIdx + threadDimension;

        firstElementIdxVec[dimIndex] = firstElementIdx;
        endElementIdxUncutVec[dimIndex] = endElementIdxUncut;
      }

      // Return element indexes, shifted by elementIdxShift.
      return {firstElementIdxVec, endElementIdxUncutVec};
    }

    /*
     * Computes the range of the elements indexes, local to the block.
     * Truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim>
    ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_block_truncated(
        const T_Acc& acc, const Vec<T_Dim>& maxNumberOfElements, const Vec<T_Dim>& elementIdxShift) {
      // Check dimension
      static_assert(alpaka::Dim<T_Acc>::value == T_Dim::value,
                    "Accelerator and maxNumberOfElements need to have same dimension.");
      auto&& [firstElementIdxLocalVec, endElementIdxLocalVec] = element_index_range_in_block(acc, elementIdxShift);

      // Truncate
      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        endElementIdxLocalVec[dimIndex] = std::min(endElementIdxLocalVec[dimIndex], maxNumberOfElements[dimIndex]);
      }

      // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
      return {firstElementIdxLocalVec, endElementIdxLocalVec};
    }

    /*
     * Computes the range of the elements indexes in grid.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim = alpaka::Dim<T_Acc>>
    ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_grid(const T_Acc& acc,
                                                                                Vec<T_Dim>& elementIdxShift) {
      // Loop on all grid dimensions.
      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        // Take into account the block index in grid.
        const uint32_t blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
        const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

        // Shift to get global indices in grid (instead of local to the block)
        elementIdxShift[dimIndex] += blockIdxInGrid * blockDimension;
      }

      // Return element indexes, shifted by elementIdxShift.
      return element_index_range_in_block(acc, elementIdxShift);
    }

    /*
     * Computes the range of the elements indexes in grid.
     * Truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim>
    ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_grid_truncated(
        const T_Acc& acc, const Vec<T_Dim>& maxNumberOfElements, Vec<T_Dim>& elementIdxShift) {
      // Check dimension
      static_assert(alpaka::Dim<T_Acc>::value == T_Dim::value,
                    "Accelerator and maxNumberOfElements need to have same dimension.");
      auto&& [firstElementIdxGlobalVec, endElementIdxGlobalVec] = element_index_range_in_grid(acc, elementIdxShift);

      // Truncate
      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        endElementIdxGlobalVec[dimIndex] = std::min(endElementIdxGlobalVec[dimIndex], maxNumberOfElements[dimIndex]);
      }

      // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
      return {firstElementIdxGlobalVec, endElementIdxGlobalVec};
    }

    /*
     * Computes the range of the element(s) index(es) in grid.
     * Truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim>
    ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_grid_truncated(
        const T_Acc& acc, const Vec<T_Dim>& maxNumberOfElements) {
      Vec<T_Dim> elementIdxShift = Vec<T_Dim>::zeros();
      return element_index_range_in_grid_truncated(acc, maxNumberOfElements, elementIdxShift);
    }

    /*********************************************
     *     1D HELPERS, LOOP ON ALL CPU ELEMENTS
     ********************************************/

    /*
     * Loop on all (CPU) elements.
     * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
     * Indexes are local to the BLOCK.
     */
    template <typename T_Acc, typename Func>
    ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_block(const T_Acc& acc,
                                                                    const uint32_t maxNumberOfElements,
                                                                    const uint32_t elementIdxShift,
                                                                    const Func& func) {
      const auto& [firstElementIdx, endElementIdx] = cms::alpakatools::element_index_range_in_block_truncated(
          acc, Vec1::all(maxNumberOfElements), Vec1::all(elementIdxShift));

      for (uint32_t elementIdx = firstElementIdx[0u]; elementIdx < endElementIdx[0u]; ++elementIdx) {
        func(elementIdx);
      }
    }

    /*
     * Overload for elementIdxShift = 0
     */
    template <typename T_Acc, typename Func>
    ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_block(const T_Acc& acc,
                                                                    const uint32_t maxNumberOfElements,
                                                                    const Func& func) {
      const uint32_t elementIdxShift = 0;
      cms::alpakatools::for_each_element_in_thread_1D_index_in_block(acc, maxNumberOfElements, elementIdxShift, func);
    }

    /*
     * Loop on all (CPU) elements.
     * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
     * Indexes are expressed in GRID 'frame-of-reference'.
     */
    template <typename T_Acc, typename Func>
    ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_grid(const T_Acc& acc,
                                                                   const uint32_t maxNumberOfElements,
                                                                   uint32_t elementIdxShift,
                                                                   const Func& func) {
      // Take into account the block index in grid to compute the element indices.
      const uint32_t blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      elementIdxShift += blockIdxInGrid * blockDimension;

      for_each_element_in_thread_1D_index_in_block(acc, maxNumberOfElements, elementIdxShift, func);
    }

    /*
     * Overload for elementIdxShift = 0
     */
    template <typename T_Acc, typename Func>
    ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_grid(const T_Acc& acc,
                                                                   const uint32_t maxNumberOfElements,
                                                                   const Func& func) {
      const uint32_t elementIdxShift = 0;
      cms::alpakatools::for_each_element_in_thread_1D_index_in_grid(acc, maxNumberOfElements, elementIdxShift, func);
    }

    /******************************************************************************
     *     1D HELPERS, LOOP ON ALL CPU ELEMENTS, AND ELEMENT/THREAD STRIDED ACCESS
     ******************************************************************************/

    /*
     * (CPU) Loop on all elements + (CPU/GPU) Strided access.
     * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
     * Stride to full problem size, by BLOCK size.
     * Indexes are local to the BLOCK.
     */
    template <typename T_Acc, typename Func>
    ALPAKA_FN_ACC void for_each_element_1D_block_stride(const T_Acc& acc,
                                                        const uint32_t maxNumberOfElements,
                                                        const uint32_t elementIdxShift,
                                                        const Func& func) {
      // Get thread / element indices in block.
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
          cms::alpakatools::element_index_range_in_block(acc, Vec1::all(elementIdxShift));

      // Stride = block size.
      const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

      // Strided access.
      for (uint32_t threadIdx = firstElementIdxNoStride[0u], endElementIdx = endElementIdxNoStride[0u];
           threadIdx < maxNumberOfElements;
           threadIdx += blockDimension, endElementIdx += blockDimension) {
        // (CPU) Loop on all elements.
        for (uint32_t i = threadIdx; i < std::min(endElementIdx, maxNumberOfElements); ++i) {
          func(i);
        }
      }
    }

    /*
     * Overload for elementIdxShift = 0
     */
    template <typename T_Acc, typename Func>
    ALPAKA_FN_ACC void for_each_element_1D_block_stride(const T_Acc& acc,
                                                        const uint32_t maxNumberOfElements,
                                                        const Func& func) {
      const uint32_t elementIdxShift = 0;
      cms::alpakatools::for_each_element_1D_block_stride(acc, maxNumberOfElements, elementIdxShift, func);
    }

    /*
     * (CPU) Loop on all elements + (CPU/GPU) Strided access.
     * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
     * Stride to full problem size, by GRID size.
     * Indexes are local to the GRID.
     */
    template <typename T_Acc, typename Func>
    ALPAKA_FN_ACC void for_each_element_1D_grid_stride(const T_Acc& acc,
                                                       const uint32_t maxNumberOfElements,
                                                       const uint32_t elementIdxShift,
                                                       const Func& func) {
      Vec1 elementIdxShiftVec = Vec1::all(elementIdxShift);

      // Get thread / element indices in block.
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
          cms::alpakatools::element_index_range_in_grid(acc, elementIdxShiftVec);

      // Stride = grid size.
      const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u]);

      // Strided access.
      for (uint32_t threadIdx = firstElementIdxNoStride[0u], endElementIdx = endElementIdxNoStride[0u];
           threadIdx < maxNumberOfElements;
           threadIdx += gridDimension, endElementIdx += gridDimension) {
        // (CPU) Loop on all elements.
        for (uint32_t i = threadIdx; i < std::min(endElementIdx, maxNumberOfElements); ++i) {
          func(i);
        }
      }
    }

    /*
     * Overload for elementIdxShift = 0
     */
    template <typename T_Acc, typename Func>
    ALPAKA_FN_ACC void for_each_element_1D_grid_stride(const T_Acc& acc,
                                                       const uint32_t maxNumberOfElements,
                                                       const Func& func) {
      const uint32_t elementIdxShift = 0;
      cms::alpakatools::for_each_element_1D_grid_stride(acc, maxNumberOfElements, elementIdxShift, func);
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAWORKDIVHELPER_H
