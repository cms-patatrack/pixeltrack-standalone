#include <iostream>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"

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
  const DevAcc1 device(alpaka::getDevByIdx<PltfAcc1>(0u));
  Queue queue(device);

  // Prepare 1D workDiv
  const Vec1& blocksPerGrid(Vec1::all(1u));
  const Vec1& threadsPerBlockOrElementsPerThread(Vec1(4u));
  const WorkDiv1& workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1>(workDiv, Print()));
  alpaka::wait(queue);
  return 0;
}
