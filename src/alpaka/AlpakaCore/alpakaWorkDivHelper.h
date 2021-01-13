#ifndef ALPAKAWORKDIVHELPER_H
#define ALPAKAWORKDIVHELPER_H

//#include <utility>

#include "AlpakaCore/alpakaConfig.h"

using namespace alpaka_common;

namespace cms {
  namespace Alpaka { 

    /*
     * Creates the accelerator-dependent workdiv.
     */
    template <typename T_Dim>
      WorkDiv<T_Dim> make_workdiv(const Vec<T_Dim>& blocksPerGrid, const Vec<T_Dim>& threadsPerBlockOrElementsPerThread) {

      // On the GPU: 
      // threadsPerBlockOrElementsPerThread is the number of threads per block.
      // Each thread is looking at a single element: elementsPerThread is always 1.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      const Vec<T_Dim>& elementsPerThread = Vec<T_Dim>::all(1);
      return WorkDiv<T_Dim>(blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
#else
      // On the CPU:
      // Run serially with a single thread per block: threadsPerBlock is always 1.
      // threadsPerBlockOrElementsPerThread is the number of elements per thread.
      const Vec<T_Dim>& threadsPerBlock = Vec<T_Dim>::all(1);
      return WorkDiv<T_Dim>(blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
#endif
    }

      /*
       * Computes the range of the element(s) global index(es) in grid.
       */
      template <typename T_Acc>
	ALPAKA_FN_ACC std::pair<uint32_t, uint32_t> element_global_index_range(const T_Acc& acc, const uint32_t maxNumberOfElements) {
      // Global thread index in grid
      const uint32_t threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
      const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

      // Global element index in grid
      // Obviously relevant for CPU only.
      // For GPU, threadDimension = 1, and firstElementIdxGlobal = endElementIdxGlobal = threadIndexGlobal.
      const uint32_t firstElementIdxGlobal = threadIdxGlobal * threadDimension;
      const uint32_t endElementIdxGlobalUncut = firstElementIdxGlobal + threadDimension;
      const uint32_t endElementIdxGlobal = std::min(endElementIdxGlobalUncut, maxNumberOfElements);

      return {firstElementIdxGlobal, endElementIdxGlobal};
    }

    } // namespace Alpaka
    }  // namespace cms

#endif  // ALPAKAWORKDIVHELPER_H
