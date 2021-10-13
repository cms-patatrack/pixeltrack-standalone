#include "CUDADataFormats/TrackingRecHit2DHostSOAView.h"

TrackingRecHit2DHostSOAView::TrackingRecHit2DHostSOAView():
  hitsStore_(hits_h.get(), 0 /* size */, 1 /* byte alignement */)
{}

void TrackingRecHit2DHostSOAView::reset() {
  hits_h.reset();
  hitsStore_.~HitsStore();
}

TrackingRecHit2DHostSOAView::TrackingRecHit2DHostSOAView(size_t size, cudaStream_t stream):
  hits_h(cms::cuda::make_host_unique<std::byte[]>(TrackingRecHit2DSOAStore::HitsStore::computeDataSize(size), stream)), 
  hitsStore_(hits_h.get(), size, 1 /* byte alignement */)
{}
