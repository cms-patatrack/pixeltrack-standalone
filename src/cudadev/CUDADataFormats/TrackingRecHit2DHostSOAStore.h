
#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAHostStore_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAHostStore_h

#include "CUDADataFormats/TrackingRecHit2DSOAView.h"
#include "CUDACore/host_unique_ptr.h"

template <typename Traits>
class TrackingRecHit2DHeterogeneous;

struct TrackingRecHit2DHostSOAStore {
  template <typename Traits>
  friend class TrackingRecHit2DHeterogeneous;

public:
  TrackingRecHit2DHostSOAStore();
  void reset();
  __device__ __forceinline__ const auto operator[](size_t i) const { return hitsView_[i]; }
  __device__ __forceinline__ size_t size() {
    return /* TODO: move to view when view will embed size */ hitsLayout_.soaMetadata().size();
  }

private:
  TrackingRecHit2DHostSOAStore(size_t size, cudaStream_t stream);
  cms::cuda::host::unique_ptr<std::byte[]> hits_h;
  TrackingRecHit2DSOAStore::HitsLayout hitsLayout_;
  TrackingRecHit2DSOAStore::HitsView hitsView_;
};

#endif  // ndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAHostStore_h