#include "CUDACore/cudaCompat.h"

namespace cms {
  namespace cudacompat {
    thread_local dim3 blockIdx;
    thread_local dim3 gridDim;
  }  // namespace cudacompat
}  // namespace cms

namespace {
  struct InitGrid {
    InitGrid() { 
      cms::cudacompat::blockIdx = {0, 0, 0};
      cms::cudacompat::gridDim = {1, 1, 1}; 
      }
  };

  const InitGrid initGrid;

}  // namespace
