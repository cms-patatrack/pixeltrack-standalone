#ifndef HeterogeneousCore_AlpakaCore_EDContext_h
#define HeterogeneousCore_AlpakaCore_EDContext_h

#include <memory>

#include "AlpakaCore/FwkContextBase.h"
#include "AlpakaCore/ProductMetadata.h"
#include "DataFormats/Product.h"
#include "Framework/WaitingTaskWithArenaHolder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class EDContext : public impl::FwkContextBase {
  public:
    // For ExternalWork module
    // In acquire()
    EDContext(edm::StreamID stream, edm::WaitingTaskWithArenaHolder holder) : FwkContextBase(stream), waitingTaskHolder_(std::move(holder)) {}
    // In produce()
    EDContext(std::shared_ptr<ProductMetadata> metadata) : FwkContextBase(std::move(metadata)) {}

    // For non-ExternalWork module
    EDContext(edm::StreamID stream) : FwkContextBase(stream) {}

    EDContext(EDContext const&) = delete;
    EDContext& operator=(EDContext const&) = delete;
    EDContext(EDContext&&) = delete;
    EDContext& operator=(EDContext&&) = delete;

    template <typename T>
    T const& get(edm::Product<T> const& data) {
      reuseAndSynchronizeQueue(data);
      return data.data();
    }

    std::shared_ptr<ProductMetadata> finishAcquire();

    void finishProduce();

  private:
    void reuseAndSynchronizeQueue(edm::ProductBase const& data);

    edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
  };
}

#endif
