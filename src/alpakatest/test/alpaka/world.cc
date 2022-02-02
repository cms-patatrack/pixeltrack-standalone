#include <iostream>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDiv.h"

namespace {
  struct Print {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc) const {
      uint32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const elemDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
      printf("Alpaka kernel thread index %u, number of elements %u\n", blockThreadIdx, elemDimension);
    }
  };
}  // namespace

int main() {
  std::cout << "World" << std::endl;

  using namespace ALPAKA_ACCELERATOR_NAMESPACE;
  const Device device(alpaka::getDevByIdx<Platform>(0u));
  Queue queue(device);

  // Prepare 1D workDiv
  const Vec1D& blocksPerGrid(Vec1D::all(1u));
  const Vec1D& threadsPerBlockOrElementsPerThread(Vec1D(4u));
  const WorkDiv1D& workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, Print()));
  alpaka::wait(queue);
  return 0;
}
