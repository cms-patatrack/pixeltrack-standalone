#ifndef ALPAKAWORKDIVHELPER_H
#define ALPAKAWORKDIVHELPER_H

//#include <utility>

#include "AlpakaCore/alpakaConfig.h"

using namespace alpaka_common;

namespace cms {
  namespace Alpaka { 


    template <typename T_Dim>
      WorkDiv<T_Dim> make_workdiv(const Vec<T_Dim>& blocksPerGrid, const Vec<T_Dim>& threadsPerBlockOrElementsPerThread) {
      //const Vec& blocksPerGrid(Vec::all(numBlocksPerGrid));
      //const Vec& threadsPerBlockOrElementsPerThread(Vec::all(numThreadsPerBlockOrElementsPerThread));

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      const Vec<T_Dim>& elementsPerThread = Vec<T_Dim>::all(1);
      return WorkDiv<T_Dim>(blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
#else
      const Vec<T_Dim>& threadsPerBlock = Vec<T_Dim>::all(1);
      return WorkDiv<T_Dim>(blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
#endif
    }









    /*
    // on the GPU, run with 32 threads in parallel per block, each looking at a single element
    // on the CPU, run serially with a single thread per block, over 32 elements
    WorkDiv make_1D_workdiv(int numBlocksPerGrid, int numThreadsPerBlockOrElementsPerThread) {
      const Vec& blocksPerGrid(Vec::all(numBlocksPerGrid));
      const Vec& threadsPerBlockOrElementsPerThread(Vec::all(numThreadsPerBlockOrElementsPerThread));

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      const Vec& elementsPerThread = Vec::all(1);
      return WorkDiv(blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
#else
      const Vec& threadsPerBlock = Vec::all(1);
      return WorkDiv(blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
#endif
}*/


    /*
    alpaka_common::WorkDiv2 make_2D_workdiv(int numBlocksPerGridX, int numBlocksPerGridY, int numThreadsPerBlockOrElementsPerThreadX, int numThreadsPerBlockOrElementsPerThreadY) {
      const Vec2 blocksPerGrid(numBlocksPerGridX, numBlocksPerGridY);
      const Vec2 threadsPerBlockOrElementsPerThread(numThreadsPerBlockOrElementsPerThreadX, numThreadsPerBlockOrElementsPerThreadY);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      const Vec2 elementsPerThread = Vec2::all(1);
      return WorkDiv2(blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
#else
      const Vec2 threadsPerBlock = Vec2::all(1);
      return WorkDiv2(blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
#endif
}*/









      /*

// Prepare 2D workDiv
    Vec2 elementsPerThread2(1u, 1u);
    const unsigned int threadsPerBlockSide = (NUM_VALUES < 32 ? NUM_VALUES : 32u);
    Vec2 threadsPerBlock2(threadsPerBlockSide, threadsPerBlockSide);
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED || \
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_BT_OMP4_ENABLED
    // on the GPU, run with 32 threads in parallel per block, each looking at a single element
    // on the CPU, run serially with a single thread per block, over 32 elements
    std::swap(threadsPerBlock2, elementsPerThread2);
#endif
    const unsigned int blocksPerGridSide = (NUM_VALUES <= 32 ? 1 : std::ceil(NUM_VALUES / 32.));
    const Vec2 blocksPerGrid2(blocksPerGridSide, blocksPerGridSide);
    const WorkDiv2 workDiv2(blocksPerGrid2, threadsPerBlock2, elementsPerThread2);

      */



















  /*
    template <typename T, typename T_Acc>
      std::pair<T, T> make_global_index_range_max(const T_Acc& acc, T max) {
      // Global thread index in grid
      const T threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
      const T threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

      // Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
      const T firstElementIdxGlobal = threadIdxGlobal * threadDimension;
      const T endElementIdxGlobalUncut = firstElementIdxGlobal + threadDimension;
      const T endElementIdxGlobal = std::min(endElementIdxGlobalUncut, max);

      return {firstElementIdxGlobal, endElementIdxGlobal};
      }*/





 } // namespace Alpaka
}  // namespace cms

#endif  // ALPAKAWORKDIVHELPER_H
