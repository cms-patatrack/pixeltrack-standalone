#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/alpakaConfig.h"

#include "alpakaAlgo1.h"



namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer : public edm::EDProducer {
  public:
    explicit TestProducer(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
    /*#ifdef TODO
    edm::EDPutTokenT<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>> putToken_;
    #endif*/
    edm::EDPutTokenT<alpaka::mem::buf::Buf<Acc2, float, Dim2, Idx>> putToken_;
    //edm::EDPutTokenT<float*> putToken_;
  };

    TestProducer::TestProducer(edm::ProductRegistry& reg)
      : rawGetToken_(reg.consumes<FEDRawDataCollection>()),
	/*#ifdef TODO
	  putToken_(reg.produces<cms::cuda::Product<cms::cuda::device::unique_ptr<float[]>>>())
	  #endif*/
        putToken_(reg.produces<alpaka::mem::buf::Buf<Acc2, float, Dim2, Idx>>())
	//putToken_(reg.produces<float*>())
  {
  }

  void TestProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    auto const value = event.get(rawGetToken_).FEDData(1200).size();
    std::cout << "TestProducer  Event " << event.eventID() << " stream " << event.streamID() << " ES int "
              << eventSetup.get<int>() << " FED 1200 size " << value << std::endl;

    /*#ifdef TODO
      cms::cuda::ScopedContextProduce ctx(event.streamID());

      ctx.emplace(event, putToken_, gpuAlgo1(ctx.stream()));
      #endif*/
    //alpakaAlgo1();

    //alpaka::mem::buf::Buf<Acc2, float, Dim2, Idx> result = alpakaAlgo1();
    //std::cout << alpaka::mem::view::getPtrNative(result) << std::endl;

    event.emplace(putToken_, alpakaAlgo1());

  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer);
