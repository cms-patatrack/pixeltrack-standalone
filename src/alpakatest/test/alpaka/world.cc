#include <iostream>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDiv.h"
#include "AlpakaCore/alpakaDevices.h"

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

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace cms::alpakatools;

int main() {
  std::cout << "World" << std::endl;

  const Device device(alpaka::getDevByIdx(*platform<Platform>, 0u));
  Queue queue(device);

  // prepare a 1D work division
  const auto blocksPerGrid = Vec1D::all(1u);
  const auto threadsPerBlockOrElementsPerThread = Vec1D(4u);
  const auto workDiv = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, Print()));
  alpaka::wait(queue);

  return 0;
}
