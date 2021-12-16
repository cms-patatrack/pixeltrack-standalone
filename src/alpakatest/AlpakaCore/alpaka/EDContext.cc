#include "AlpakaCore/EDContext.h"
#include "AlpakaCore/alpakaConfigFwd.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  std::shared_ptr<ProductMetadata> EDContext::finishAcquire() {
    metadataPtr()->enqueueCallback(std::move(waitingTaskHolder_));
    return releaseMetadataPtr();
  }

  void EDContext::finishProduce() {
    metadataPtr()->recordEvent();
  }

  void EDContext::reuseAndSynchronizeQueue(edm::ProductBase const& data) {
    auto const& metadata = *data.metadata<std::shared_ptr<ProductMetadata>>();
    if (not queueUsed()) {
      if (auto queue = metadata.tryReuseQueue()) {
        setQueue(queue);
      }
    }
    metadata.synchronizeWith(queue());
  }  
}
