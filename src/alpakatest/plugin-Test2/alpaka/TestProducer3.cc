#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/alpakaConfig.h"


namespace {
  constexpr unsigned int NUM_VALUES = 1000;

  struct TestPrint {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc,
                                  const T_Data* __restrict__ a,
                                  unsigned int numElements) const {
      // Global thread index in Dim2 grid
      const auto& threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
      const uint32_t threadIdxGlobalX(threadIdxGlobal[0u]);
      const uint32_t threadIdxGlobalY(threadIdxGlobal[1u]);

      // Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
      const auto& threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
      const uint32_t threadDimensionX(threadDimension[0u]);
      const uint32_t firstElementIdxGlobalX = threadIdxGlobalX * threadDimensionX;
      const uint32_t endElementIdxGlobalXUncut = firstElementIdxGlobalX + threadDimensionX;
      const uint32_t endElementIdxGlobalX = std::min(endElementIdxGlobalXUncut, numElements);

      const uint32_t threadDimensionY(threadDimension[1u]);
      const uint32_t firstElementIdxGlobalY = threadIdxGlobalY * threadDimensionY;
      const uint32_t endElementIdxGlobalYUncut = firstElementIdxGlobalY + threadDimensionY;
      const uint32_t endElementIdxGlobalY = std::min(endElementIdxGlobalYUncut, numElements);

      for (uint32_t col = firstElementIdxGlobalX; col < endElementIdxGlobalX; ++col) {
        for (uint32_t row = firstElementIdxGlobalY; row < endElementIdxGlobalY; ++row) {
          printf("TestPrint, row = %u, col = %u, a[row,col] = %f.\n",
		 row,
		 col,
		 a[row * numElements + col]);
        }
      }
    }
  };

}



namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer3 : public edm::EDProducer {
  public:
    explicit TestProducer3(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    /*#ifdef TODO
    edm::EDGetTokenT<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>> getToken_;
    #endif*/
    edm::EDGetTokenT<alpaka::mem::buf::Buf<Acc2, float, Dim2, Idx>> getToken_;
  };

  TestProducer3::TestProducer3(edm::ProductRegistry& reg) 
    :
    /*#ifdef TODO
      getToken_(reg.consumes<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>>())
      #endif*/
    getToken_(reg.consumes<alpaka::mem::buf::Buf<Acc2, float, Dim2, Idx>>())
  {
  }

  void TestProducer3::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    /*#ifdef TODO
    auto const& tmp = event.get(getToken_); 
    cms::cuda::ScopedContextProduce ctx(tmp);
    #endif*/


    auto const result = event.get(getToken_);
    std::cout << "TestProducer3 Event " << event.eventID() << " stream " << event.streamID() << std::endl;





    const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    const DevAcc2 device(alpaka::pltf::getDevByIdx<PltfAcc2>(0u));
    const Vec size(NUM_VALUES);
    Queue queue(device);

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

    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<Acc2>(workDiv2,
								  TestPrint(),
								  alpaka::mem::view::getPtrNative(result),
								  NUM_VALUES));
    alpaka::wait::wait(queue);








  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer3);
