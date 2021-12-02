#include "CUDADataFormats/TrackingRecHit2DHostSOAStore.h"

TrackingRecHit2DHostSOAStore::TrackingRecHit2DHostSOAStore():
  hitsStore_(hits_h.get(), 0 /* size */, 1 /* byte alignement */)
{}

void TrackingRecHit2DHostSOAStore::reset() {
  hits_h.reset();
  hitsStore_ = TrackingRecHit2DSOAStore::HitsStore();
}

TrackingRecHit2DHostSOAStore::TrackingRecHit2DHostSOAStore(size_t size, cudaStream_t stream):
  hits_h(cms::cuda::make_host_unique<std::byte[]>(TrackingRecHit2DSOAStore::HitsStore::computeDataSize(size), stream)), 
  hitsStore_(hits_h.get(), size)
{}
