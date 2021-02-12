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
        // For GPU, threadDimension = 1, and endElementIdxGlobal = firstElementIdxLocal + 1.
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
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_global_index_range_truncated(const T_Acc& acc, const Vec<T_Dim>& maxNumberOfElements) {
      static_assert(alpaka::dim::Dim<T_Acc>::value == T_Dim::value,
                    "Accelerator and maxNumberOfElements need to have same dimension.");
      auto&& [firstElementIdxGlobalVec, endElementIdxGlobalVec] = element_global_index_range(acc);

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        endElementIdxGlobalVec[dimIndex] = std::min(endElementIdxGlobalVec[dimIndex], maxNumberOfElements[dimIndex]);
      }

      return {firstElementIdxGlobalVec, endElementIdxGlobalVec};
    }


    /*
     * Computes the range of the element(s) index(es) local to the block.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_local_index_range(const T_Acc& acc) {
      Vec<T_Dim> firstElementIdxLocalVec = Vec<T_Dim>::zeros();
      Vec<T_Dim> endElementIdxUncutLocalVec = Vec<T_Dim>::zeros();

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        // Local thread index in block (along dimension dimIndex).
        const uint32_t threadIdxLocal(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndex]);
        const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[dimIndex]);

        // Local element index in block (along dimension dimIndex).
        // Obviously relevant for CPU only.
        // For GPU, threadDimension = 1, and endElementIdxGlobal = firstElementIdxLocal + 1.
        const uint32_t firstElementIdxLocal = threadIdxLocal * threadDimension;
        const uint32_t endElementIdxUncutLocal = firstElementIdxLocal + threadDimension;

        firstElementIdxLocalVec[dimIndex] = firstElementIdxLocal;
        endElementIdxUncutLocalVec[dimIndex] = endElementIdxUncutLocal;
      }

      return {firstElementIdxLocalVec, endElementIdxUncutLocalVec};
    }

    /*
     * Computes the range of the element(s) global index(es) in grid.
     * Truncated by the max number of elements of interest.
     */
    template <typename T_Acc, typename T_Dim>
      ALPAKA_FN_ACC std::pair<Vec<T_Dim>, Vec<T_Dim>> element_local_index_range_truncated(const T_Acc& acc, const Vec<T_Dim>& maxNumberOfElements) {
      static_assert(alpaka::dim::Dim<T_Acc>::value == T_Dim::value,
                    "Accelerator and maxNumberOfElements need to have same dimension.");
      auto&& [firstElementIdxLocalVec, endElementIdxLocalVec] = element_local_index_range(acc);

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
        endElementIdxLocalVec[dimIndex] = std::min(endElementIdxLocalVec[dimIndex], maxNumberOfElements[dimIndex]);
      }

      return {firstElementIdxLocalVec, endElementIdxLocalVec};
    }


    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_global_index(const T_Acc& acc, const Vec<T_Dim>& problemSize, const Vec<T_Dim>& indexShift, Func func) {

      const auto& [firstElementIdxGlobal, endElementIdxGlobal] =
      cms::alpakatools::element_global_index_range_truncated(acc, problemSize);

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
	const uint32_t firstElementIdx = firstElementIdxGlobal[dimIndex] + indexShift[dimIndex];
	const uint32_t endElementIdx = endElementIdxGlobal[dimIndex] + indexShift[dimIndex];

	for (uint32_t elementIdx = firstElementIdx; elementIdx < endElementIdx; ++elementIdx) {
	  func(elementIdx);
	}
      }

    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_local_index(const T_Acc& acc, const Vec<T_Dim>& problemSize, const Vec<T_Dim>& indexShift, Func func) {

      const auto& [firstElementIdxLocal, endElementIdxLocal] =
      cms::alpakatools::element_local_index_range_truncated(acc, problemSize);

      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
	const uint32_t firstElementIdx = firstElementIdxLocal[dimIndex] + indexShift[dimIndex];
	const uint32_t endElementIdx = endElementIdxLocal[dimIndex] + indexShift[dimIndex];

	for (uint32_t elementIdx = firstElementIdx; elementIdx < endElementIdx; ++elementIdx) {
	  func(elementIdx);
	}
      }

    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_grid_stride(const T_Acc& acc, const Vec<T_Dim>& problemSize, const Vec<T_Dim>& indexShift, Func func) {
      
      const auto &[firstElementIdxNoStrideNoShift, endElementIdxNoStrideNoShift] =
      cms::alpakatools::element_global_index_range(acc);
      
      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
	const uint32_t gridDimension(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[dimIndex]);
	const uint32_t firstElementIdxNoStride = firstElementIdxNoStrideNoShift[dimIndex] + indexShift[dimIndex];
	const uint32_t endElementIdxNoStride = endElementIdxNoStrideNoShift[dimIndex] + indexShift[dimIndex];

	for (uint32_t threadIdx = firstElementIdxNoStride, endElementIdx = endElementIdxNoStride;
	     threadIdx < problemSize[dimIndex];
	     threadIdx += gridDimension, endElementIdx += gridDimension) {
	  
	  for (uint32_t i = threadIdx; i < std::min(endElementIdx, problemSize[dimIndex]); ++i) {
	    func(i);
	  }
	}
      }
      
    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_block_stride(const T_Acc& acc, const Vec<T_Dim>& problemSize, const Vec<T_Dim>& indexShift, Func func) {

      const auto &[firstElementIdxNoStrideNoShift, endElementIdxNoStrideNoShift] =
      cms::alpakatools::element_local_index_range(acc);
      
      for (typename T_Dim::value_type dimIndex(0); dimIndex < T_Dim::value; ++dimIndex) {
	const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);
	const uint32_t firstElementIdxNoStride = firstElementIdxNoStrideNoShift[dimIndex] + indexShift[dimIndex];
	const uint32_t endElementIdxNoStride = endElementIdxNoStrideNoShift[dimIndex] + indexShift[dimIndex];

	for (uint32_t threadIdx = firstElementIdxNoStride, endElementIdx = endElementIdxNoStride;
	     threadIdx < problemSize[dimIndex];
	     threadIdx += blockDimension, endElementIdx += blockDimension) {
	  
	  for (uint32_t i = threadIdx; i < std::min(endElementIdx, problemSize[dimIndex]); ++i) {
	    func(i);
	  }
	}
      }
      
    }


    // 1D HELPERS    

    // LOOP ON ELEMENTS ONLY, GLOBAL INDICES (per grid)
    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_global_index(const T_Acc& acc, const T_Dim problemSize, T_Dim indexShift, Func func) {     
      cms::alpakatools::for_each_element_in_thread_global_index(acc, Vec1::all(problemSize), Vec1::all(indexShift), func);
    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_global_index(const T_Acc& acc, const T_Dim problemSize, Func func) {
      T_Dim indexShift = 0; 
      cms::alpakatools::for_each_element_in_thread_1D_global_index(acc, problemSize, indexShift, func);
    }

    // LOOP ON ELEMENTS ONLY, LOCAL INDICES (per block)
    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_local_index(const T_Acc& acc, const T_Dim problemSize, T_Dim indexShift, Func func) {     
      cms::alpakatools::for_each_element_in_thread_local_index(acc, Vec1::all(problemSize), Vec1::all(indexShift), func);
    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_in_thread_1D_local_index(const T_Acc& acc, const T_Dim problemSize, Func func) {
      T_Dim indexShift = 0; 
      cms::alpakatools::for_each_element_in_thread_1D_local_index(acc, problemSize, indexShift, func);
    }

    

    // HANDLES THREAD AND ELEMENTS INDICES FOR STRIDED ACCESS, GLOBAL INDICES (per grid)
    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_1D_grid_stride(const T_Acc& acc, const T_Dim problemSize, T_Dim indexShift, Func func) {     
      cms::alpakatools::for_each_element_grid_stride(acc, Vec1::all(problemSize), Vec1::all(indexShift), func);
    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_1D_grid_stride(const T_Acc& acc, const T_Dim problemSize, Func func) { 
      T_Dim indexShift = 0; 
      cms::alpakatools::for_each_element_1D_grid_stride(acc, problemSize, indexShift, func);
    }

    // HANDLES THREAD AND ELEMENTS INDICES FOR STRIDED ACCESS, LOCAL INDICES (per block)
    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_1D_block_stride(const T_Acc& acc, const T_Dim problemSize, T_Dim indexShift, Func func) {     
      cms::alpakatools::for_each_element_block_stride(acc, Vec1::all(problemSize), Vec1::all(indexShift), func);
    }

    template <typename T_Acc, typename T_Dim = alpaka::dim::Dim<T_Acc>, typename Func>
      ALPAKA_FN_ACC void for_each_element_1D_block_stride(const T_Acc& acc, const T_Dim problemSize, Func func) {
      T_Dim indexShift = 0;      
      cms::alpakatools::for_each_element_1D_block_stride(acc, problemSize, indexShift, func);
    }

    
  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAWORKDIVHELPER_H
