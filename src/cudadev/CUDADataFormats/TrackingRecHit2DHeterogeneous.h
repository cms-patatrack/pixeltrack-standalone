#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "CUDADataFormats/TrackingRecHit2DSOAView.h"
#include "CUDADataFormats/HeterogeneousSoA.h"
#include "CUDADataFormats/TrackingRecHit2DHostSOAStore.h"

template <typename Traits>
class TrackingRecHit2DHeterogeneous {
public:
  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using PhiBinner = TrackingRecHit2DSOAStore::PhiBinner;

  TrackingRecHit2DHeterogeneous() = default;

  explicit TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                         pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                         uint32_t const* hitsModuleStart,
                                         cudaStream_t stream);

  ~TrackingRecHit2DHeterogeneous() = default;

  TrackingRecHit2DHeterogeneous(const TrackingRecHit2DHeterogeneous&) = delete;
  TrackingRecHit2DHeterogeneous& operator=(const TrackingRecHit2DHeterogeneous&) = delete;
  TrackingRecHit2DHeterogeneous(TrackingRecHit2DHeterogeneous&&) = default;
  TrackingRecHit2DHeterogeneous& operator=(TrackingRecHit2DHeterogeneous&&) = default;

  TrackingRecHit2DSOAStore* store() { return m_store.get(); }
  TrackingRecHit2DSOAStore const* store() const { return m_store.get(); }

  auto nHits() const { return m_nHits; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  auto phiBinner() { return m_phiBinner; }
  auto phiBinnerStorage() { return m_phiBinnerStorage; }
  auto iphi() { return m_iphi; }

  // Transfer the local and global coordinates, charge and size
  TrackingRecHit2DHostSOAStore hitsToHostAsync(cudaStream_t stream) const;
  
  // apparently unused
  //cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;

private:
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious
  
  unique_ptr<TrackingRecHit2DSOAStore::PhiBinner> m_PhiBinnerStore;              //!
  unique_ptr<TrackingRecHit2DSOAStore::AverageGeometry> m_AverageGeometryStore;  //!

  unique_ptr<TrackingRecHit2DSOAStore> m_store;  //!

  uint32_t m_nHits;
  
  unique_ptr<std::byte[]> m_hitsSupportLayerStartStore;                                          //!

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  PhiBinner* m_phiBinner;
  PhiBinner::index_type* m_phiBinnerStorage;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};

#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"

template <typename Traits>
TrackingRecHit2DHeterogeneous<Traits>::TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                                                     pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                                     uint32_t const* hitsModuleStart,
                                                                     cudaStream_t stream)
    : m_nHits(nHits), m_hitsModuleStart(hitsModuleStart) {
  auto store = Traits::template make_host_unique<TrackingRecHit2DSOAStore>(stream);

  store->m_nHits = nHits;
  m_store = Traits::template make_device_unique<TrackingRecHit2DSOAStore>(stream);
  m_AverageGeometryStore = Traits::template make_device_unique<TrackingRecHit2DSOAStore::AverageGeometry>(stream);
  store->m_averageGeometry = m_AverageGeometryStore.get();
  store->m_cpeParams = cpeParams;
  store->m_hitsModuleStart = hitsModuleStart;

  // if empty do not bother
  if (0 == nHits) {
    if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
      cms::cuda::copyAsync(m_store, store, stream);
    } else {
      m_store.reset(store.release());  // NOLINT: std::move() breaks CUDA version
    }
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated
  //m_store16 = Traits::template make_device_unique<uint16_t[]>(nHits * n16, stream);
  //m_store32 =
  //    Traits::template make_device_unique<float[]>(nHits * n32 + phase1PixelTopology::numberOfLayers + 1, stream);
  // We need to store all SoA rows for TrackingRecHit2DSOAView::HitsView(nHits) + 
  // (phase1PixelTopology::numberOfLayers + 1) TrackingRecHit2DSOAView::PhiBinner::index_type.
  // As mentioned above, alignment is not important, yet we want to have 32 bits 
  // (TrackingRecHit2DSOAView::PhiBinner::index_type exactly) alignement for the second part.
  // In order to simplify code, we align all to the minimum necessary size (sizeof(TrackingRecHit2DSOAStore::PhiBinner::index_type)).
  {
    // Simplify a bit following computations
    const size_t align = sizeof(TrackingRecHit2DSOAStore::PhiBinner::index_type);
    const size_t phiBinnerByteSize =
      (phase1PixelTopology::numberOfLayers + 1) * sizeof (TrackingRecHit2DSOAStore::PhiBinner::index_type);
    // Allocate the buffer
    m_hitsSupportLayerStartStore = Traits::template make_device_unique<std::byte[]> (
      TrackingRecHit2DSOAStore::HitsStore::computeDataSize(m_nHits, align) +
        TrackingRecHit2DSOAStore::SupportObjectsStore::computeDataSize(m_nHits, align) +
        phiBinnerByteSize, 
      stream);
    // Split the buffer in stores and array
    store->m_hitsStore.~HitsStore();
    new (&store->m_hitsStore) TrackingRecHit2DSOAStore::HitsStore(m_hitsSupportLayerStartStore.get(), nHits, align);
    store->m_supportObjectsStore.~SupportObjectsStore();
    new (&store->m_supportObjectsStore) TrackingRecHit2DSOAStore::SupportObjectsStore(store->m_hitsStore.soaMetadata().nextByte(), nHits, 1);
    m_hitsLayerStart = store->m_hitsLayerStart = reinterpret_cast<uint32_t *> (store->m_supportObjectsStore.soaMetadata().nextByte());
    // Record additional references
    store->m_hitsAndSupportView.~HitsAndSupportView();
    new (&store->m_hitsAndSupportView) TrackingRecHit2DSOAStore::HitsAndSupportView(
      store->m_hitsStore,
      store->m_supportObjectsStore
    );
    m_phiBinnerStorage = store->m_phiBinnerStorage = store->m_supportObjectsStore.phiBinnerStorage();
    m_iphi = store->m_supportObjectsStore.iphi();
  }
  m_PhiBinnerStore = Traits::template make_device_unique<TrackingRecHit2DSOAStore::PhiBinner>(stream);

  static_assert(sizeof(TrackingRecHit2DSOAStore::hindex_type) == sizeof(float));
  static_assert(sizeof(TrackingRecHit2DSOAStore::hindex_type) == sizeof(TrackingRecHit2DSOAStore::PhiBinner::index_type));

  // copy all the pointers
  m_phiBinner = store->m_phiBinner = m_PhiBinnerStore.get();
  
  // transfer view
  if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
    cms::cuda::copyAsync(m_store, store, stream);
  } else {
    m_store.reset(store.release());  // NOLINT: std::move() breaks CUDA version
  }
}

using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits>;
using TrackingRecHit2DCUDA = TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits>;
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::CPUTraits>;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous<cms::cudacompat::HostTraits>;

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h