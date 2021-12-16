#include "AlpakaCore/Context.h"
#include "AlpakaCore/EDProducer.h"
#include "AlpakaCore/EDContext.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  void EDProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    EDContext ctxImpl(event.streamID());
    Event ev(event, ctxImpl);
    Context ctx = ctxImpl.makeContext();
    produce(ev, eventSetup, ctx);
    ctxImpl.finishProduce();
  }

  ////////////////////

  void EDProducerExternalWork::acquire(edm::Event const& event, edm::EventSetup const& eventSetup, edm::WaitingTaskWithArenaHolder holder) {
    EDContext ctxImpl(event.streamID(), std::move(holder));
    Event const ev(event, ctxImpl);
    Context ctx = ctxImpl.makeContext();
    acquire(ev, eventSetup, ctx);
    metadata_ = ctxImpl.finishAcquire();
  }

  void EDProducerExternalWork::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    EDContext ctxImpl(std::move(metadata_));
    Event ev(event, ctxImpl);
    Context ctx = ctxImpl.makeContext();
    produce(ev, eventSetup, ctx);
    ctxImpl.finishProduce();
  }

}
