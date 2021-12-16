#ifndef HeterogeneousCore_AlpakaCore_stream_EDProducer_h
#define HeterogeneousCore_AlpakaCore_stream_EDProducer_h

#include "AlpakaCore/alpakaConfigFwd.h"
#include "AlpakaCore/Event.h"
#include "DataFormats/Product.h"
#include "Framework/EDProducer.h"
#include "Framework/Host.h"
#include "Framework/ProductEDGetToken.h"
#include "Framework/ProductEDPutToken.h"
#include "Framework/ProductRegistry.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class Context;

  namespace impl {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
    template <typename T>
    struct ConsumesHelper {
      static edm::EDGetTokenT<T> consumes(edm::ProductRegistry& reg) {
        return reg.consumes<T>();
      }
    };

    template <typename T>
    struct ProducesHelper {
      static edm::EDPutTokenT<T> produces(edm::ProductRegistry& reg) {
        return reg.produces<T>();
      }
    };

#else // asynchronous backends

    template <typename T>
    struct ConsumesHelper {
      static edm::EDGetTokenT<T> consumes(edm::ProductRegistry& reg) {
        return edm::ProductEDGetToken<T>::toToken(reg.consumes<edm::Product<T>>());
      }
    };

    template <typename T>
    struct ProducesHelper {
      static edm::EDPutTokenT<T> produces(edm::ProductRegistry& reg) {
        return edm::ProductEDPutToken<T>::toToken(reg.produces<edm::Product<T>>());
      }
    };
#endif

    // Specializations for host products are the same
    template <typename T>
    struct ConsumesHelper<edm::Host<T>> {
      static edm::EDGetTokenT<edm::Host<T>> consumes(edm::ProductRegistry& reg) {
        return edm::ProductEDGetToken<T>::toHostToken(reg.consumes<T>());
      }
    };

    template <typename T>
    struct ProducesHelper<edm::Host<T>> {
      static edm::EDGetTokenT<edm::Host<T>> produces(edm::ProductRegistry& reg) {
        return edm::ProductEDPutToken<T>::toHostToken(reg.produces<T>());
      }
    };
  }

  class EDProducer : public edm::EDProducer {
  public:
    EDProducer(edm::ProductRegistry& reg) : reg_(&reg) {}
    ~EDProducer() override = default;

    template <typename T>
    edm::EDGetTokenT<T> consumes() {
      return impl::ConsumesHelper<T>::consumes(*reg_);
    }

    template <typename T>
    edm::EDPutTokenT<T> produces() {
      return impl::ProducesHelper<T>::produces(*reg_);
    }

    void produce(edm::Event& event, edm::EventSetup const& eventSetup) final;
    virtual void produce(Event& event, edm::EventSetup const& eventSetup, Context& ctx) = 0;

  private:
    edm::ProductRegistry *reg_;
  };

  class EDProducerExternalWork : public edm::EDProducerExternalWork {
  public:
    EDProducerExternalWork(edm::ProductRegistry& reg) : reg_(&reg) {}
    ~EDProducerExternalWork() override = default;

    template <typename T>
    edm::EDGetTokenT<T> consumes() {
      return impl::ConsumesHelper<T>::consumes(*reg_);
    }

    template <typename T>
    edm::EDPutTokenT<T> produces() {
      return impl::ProducesHelper<T>::produces(*reg_);
    }

    void acquire(edm::Event const& event, edm::EventSetup const& eventSetup, edm::WaitingTaskWithArenaHolder holder) final;
    virtual void acquire(Event const& event, edm::EventSetup const& eventSetup, Context& ctx) = 0;

    void produce(edm::Event& event, edm::EventSetup const& eventSetup) final;
    virtual void produce(Event& event, edm::EventSetup const& eventSetup, Context& ctx) = 0;

  private:
    edm::ProductRegistry *reg_;
    std::shared_ptr<ProductMetadata> metadata_;
  };
}

#endif
