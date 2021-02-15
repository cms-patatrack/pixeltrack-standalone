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

    template <typename T_Acc>
      ALPAKA_FN_ACC bool once_per_block_1D(const T_Acc& acc, uint32_t i) {
      const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      return (i % blockDimension == 0);
    }

    /*
     * Computes the range of the element(s) index(es), local to the block.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_block(const T_Acc& acc, const Vec<T_Dim>& elementIndexShift) {
      Vec<T_Dim> firstElementIdxVec = Vec<T_Dim>::zeros();
      Vec<T_Dim> endElementIdxUncutVec = Vec<T_Dim>::zeros();

      // Loop on all grid dimensions.
      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
	// Take into account the thread index in block to compute the element indices.
        const uint32_t threadIdxLocal(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndex]);
        const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[dimIndex]);

        // Local element index in block (along dimension dimIndex).
        // Obviously relevant for CPU only.
        // For GPU, threadDimension = 1, and endElementIdxGlobal = firstElementIdxLocal + 1.
        const uint32_t firstElementIdxLocal = threadIdxLocal * threadDimension;
	const uint32_t firstElementIdx = firstElementIdxLocal + elementIndexShift[dimIndex]; // Add the shift!
        const uint32_t endElementIdxUncut = firstElementIdx + threadDimension;

        firstElementIdxVec[dimIndex] = firstElementIdx;
        endElementIdxUncutVec[dimIndex] = endElementIdxUncut;
      }

      return {firstElementIdxVec, endElementIdxUncutVec};
    }

    /*
     * Computes the range of the element(s) index(es), local to the block.
     * Truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_block_truncated(const T_Acc& acc, const Vec<T_Dim>& maxNumberOfElements, const Vec<T_Dim>& elementIndexShift) {
      static_assert(alpaka::dim::Dim<T_Acc>::value == T_Dim::value,
                    "Accelerator and maxNumberOfElements need to have same dimension.");
      auto&& [firstElementIdxLocalVec, endElementIdxLocalVec] = element_index_range_in_block(acc, elementIndexShift);

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        endElementIdxLocalVec[dimIndex] = std::min(endElementIdxLocalVec[dimIndex], maxNumberOfElements[dimIndex]);
      }

      return {firstElementIdxLocalVec, endElementIdxLocalVec};
    }


    /*
     * Computes the range of the element(s) index(es) in grid.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_grid(const T_Acc& acc, Vec<T_Dim>& elementIndexShift) {
      // Loop on all grid dimensions.
      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        // Take into account the block index in grid to compute the element indices.
        const uint32_t blockIdxInGrid(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
        const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

	elementIndexShift[dimIndex] += blockIdxInGrid * blockDimension;
      }

      return element_index_range_in_block(acc, elementIndexShift);
    }

    /*
     * Computes the range of the element(s) index(es) in grid.
     * Truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_grid_truncated(const T_Acc& acc, const Vec<T_Dim>& maxNumberOfElements, Vec<T_Dim>& elementIndexShift) {
      static_assert(alpaka::dim::Dim<T_Acc>::value == T_Dim::value,
                    "Accelerator and maxNumberOfElements need to have same dimension.");
      auto&& [firstElementIdxGlobalVec, endElementIdxGlobalVec] = element_index_range_in_grid(acc, elementIndexShift);

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        endElementIdxGlobalVec[dimIndex] = std::min(endElementIdxGlobalVec[dimIndex], maxNumberOfElements[dimIndex]);
      }

      return {firstElementIdxGlobalVec, endElementIdxGlobalVec};
    }


    /*
     * Computes the range of the element(s) index(es) in grid.
     * Truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_index_range_in_grid_truncated(const T_Acc& acc, const Vec<T_Dim>& maxNumberOfElements) {
      
      Vec<T_Dim> elementIndexShift = Vec<T_Dim>::zeros();
      return element_index_range_in_grid_truncated(acc, elementIndexShift, maxNumberOfElements);
    }


    // 1D HELPERS

    template <typename T_Acc, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_block(const T_Acc& acc, const uint32_t maxNumberOfElements, const uint32_t elementIndexShift, const Func& func) {

      const auto& [firstElementIdx, endElementIdx] =
	cms::alpakatools::element_index_range_in_block_truncated(acc, Vec1::all(maxNumberOfElements), Vec1::all(elementIndexShift));

      for (uint32_t elementIdx = firstElementIdx[0u]; elementIdx < endElementIdx[0u]; ++elementIdx) {
	func(elementIdx);
      }
    }


    template <typename T_Acc, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_grid(const T_Acc& acc, const uint32_t maxNumberOfElements, uint32_t elementIndexShift, const Func& func) {

      // Take into account the block index in grid to compute the element indices.
      const uint32_t blockIdxInGrid(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      elementIndexShift += blockIdxInGrid * blockDimension;

      for_each_element_in_thread_1D_index_in_block(acc, maxNumberOfElements, elementIndexShift, func);
    }



    template <typename T_Acc, typename Func>
      ALPAKA_FN_ACC void for_each_element_1D_block_stride(const T_Acc& acc, const uint32_t maxNumberOfElements, const uint32_t elementIndexShift, const Func& func) {

      const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
	cms::alpakatools::element_index_range_in_block(acc, Vec1::all(elementIndexShift));
      
      const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

      for (uint32_t threadIdx = firstElementIdxNoStride[0u], endElementIdx = endElementIdxNoStride[0u];
	   threadIdx < maxNumberOfElements;
	   threadIdx += blockDimension, endElementIdx += blockDimension) {
	  
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, maxNumberOfElements); ++i) {
	  func(i);
	}
      }
      
    }

    
    template <typename T_Acc, typename Func>
      ALPAKA_FN_ACC void for_each_element_1D_grid_stride(const T_Acc& acc, const uint32_t maxNumberOfElements, const uint32_t elementIndexShift, const Func& func) {

      Vec1 elementIndexShiftVec = Vec1::all(elementIndexShift);
      
      const auto &[firstElementIdxNoStride, endElementIdxNoStride] =
	cms::alpakatools::element_index_range_in_grid(acc, elementIndexShiftVec);
      
      const uint32_t gridDimension(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u]);

      for (uint32_t threadIdx = firstElementIdxNoStride[0u], endElementIdx = endElementIdxNoStride[0u];
	   threadIdx < maxNumberOfElements;
	   threadIdx += gridDimension, endElementIdx += gridDimension) {
	  
	for (uint32_t i = threadIdx; i < std::min(endElementIdx, maxNumberOfElements); ++i) {
	  func(i);
	}
      }
      
    }

    


     

    // LOOP ON ELEMENTS ONLY, GLOBAL INDICES (per grid)
    /*template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_grid(const T_Acc& acc, const T_Dim maxNumberOfElements, T_Dim elementIndexShift, Func func) {     
      cms::alpakatools::for_each_element_in_thread_index_in_grid(acc, Vec1::all(maxNumberOfElements), Vec1::all(elementIndexShift), func);
      }*/

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_grid(const T_Acc& acc, const T_Dim maxNumberOfElements, const Func& func) {
      T_Dim elementIndexShift = 0; 
      cms::alpakatools::for_each_element_in_thread_1D_index_in_grid(acc, maxNumberOfElements, elementIndexShift, func);
    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_block(const T_Acc& acc, const T_Dim maxNumberOfElements, const Func& func) {
      T_Dim elementIndexShift = 0; 
      cms::alpakatools::for_each_element_in_thread_1D_index_in_block(acc, maxNumberOfElements, elementIndexShift, func);
    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_1D_grid_stride(const T_Acc& acc, const T_Dim maxNumberOfElements, const Func& func) { 
      T_Dim elementIndexShift = 0; 
      cms::alpakatools::for_each_element_1D_grid_stride(acc, maxNumberOfElements, elementIndexShift, func);
    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_1D_block_stride(const T_Acc& acc, const T_Dim maxNumberOfElements, const Func& func) {
      T_Dim elementIndexShift = 0;      
      cms::alpakatools::for_each_element_1D_block_stride(acc, maxNumberOfElements, elementIndexShift, func);
    }

    
  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAWORKDIVHELPER_H
