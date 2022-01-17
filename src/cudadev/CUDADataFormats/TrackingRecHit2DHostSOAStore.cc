#include "CUDADataFormats/TrackingRecHit2DHostSOAStore.h"

TrackingRecHit2DHostSOAStore::TrackingRecHit2DHostSOAStore()
    : hitsLayout_(hits_h.get(), 0 /* size */, 1 /* byte alignement */) {}

void TrackingRecHit2DHostSOAStore::reset() {
  hits_h.reset();
  hitsLayout_ = TrackingRecHit2DSOAStore::HitsLayout();
}

TrackingRecHit2DHostSOAStore::TrackingRecHit2DHostSOAStore(size_t size, cudaStream_t stream)
    : hits_h(cms::cuda::make_host_unique<std::byte[]>(TrackingRecHit2DSOAStore::HitsLayout::computeDataSize(size),
                                                      stream)),
      hitsLayout_(hits_h.get(), size),
      hitsView_(hitsLayout_) {}
