#include "CUDACore/cudaCompat.h"

namespace cms {
  namespace hipcompat {
    thread_local dim3 blockIdx;
    thread_local dim3 gridDim;
  }  // namespace hipcompat
}  // namespace cms

namespace {
  struct InitGrid {
    InitGrid() { cms::hipcompat::resetGrid(); }
  };

  const InitGrid initGrid;

}  // namespace
