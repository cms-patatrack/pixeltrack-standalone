#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include <cuda_runtime.h>

#include "CUDADataFormats/gpuClusteringConstants.h"
#include "CUDACore/HistoContainer.h"
#include "CUDACore/cudaCompat.h"
#include "Geometry/phase1PixelTopology.h"
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAStore {
public:
  using hindex_type = uint32_t;  // if above is <=2^32

  using PhiBinner = cms::cuda::HistoContainer<int16_t, 128, -1, 8 * sizeof(int16_t), hindex_type, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DHeterogeneous;
  
  __device__ __forceinline__ uint32_t nHits() const { return m_nHits; }

  // Our arrays do not require specific alignment as access will not be coalesced in the current implementation
  // Sill, we need the 32 bits integers to be aligned, so we simply declare the SoA with the 32 bits fields first
  // and the 16 bits behind (as they have a looser alignment requirement. Then the SoA can be create with a byte 
  // alignment of 1)
  GENERATE_SOA_LAYOUT(HitsLayoutTemplate,
    // 32 bits section
    // local coord
    SOA_COLUMN(float, xLocal),
    SOA_COLUMN(float, yLocal),
    SOA_COLUMN(float, xerrLocal),
    SOA_COLUMN(float, yerrLocal),
    
    // global coord
    SOA_COLUMN(float, xGlobal),
    SOA_COLUMN(float, yGlobal),
    SOA_COLUMN(float, zGlobal),
    SOA_COLUMN(float, rGlobal),
    // global coordinates continue in the 16 bits section

    // cluster properties
    SOA_COLUMN(int32_t, charge),
          
    // 16 bits section (and cluster properties immediately continued)
    SOA_COLUMN(int16_t, clusterSizeX),
    SOA_COLUMN(int16_t, clusterSizeY)
  )
  
  // The hits layout does not use default alignment but a more relaxed one.
  using HitsLayout = HitsLayoutTemplate<sizeof(TrackingRecHit2DSOAStore::PhiBinner::index_type)>;
  
  GENERATE_SOA_VIEW(HitsViewTemplate,
    SOA_VIEW_LAYOUT_LIST(
      SOA_VIEW_LAYOUT(HitsLayout, hitsLayout)
    ),
    SOA_VIEW_VALUE_LIST(
      SOA_VIEW_VALUE(hitsLayout, xLocal),
      SOA_VIEW_VALUE(hitsLayout, yLocal),
      SOA_VIEW_VALUE(hitsLayout, xerrLocal),
      SOA_VIEW_VALUE(hitsLayout, yerrLocal),
      
      SOA_VIEW_VALUE(hitsLayout, xGlobal),
      SOA_VIEW_VALUE(hitsLayout, yGlobal),
      SOA_VIEW_VALUE(hitsLayout, zGlobal),
      SOA_VIEW_VALUE(hitsLayout, rGlobal),
      
      SOA_VIEW_VALUE(hitsLayout, charge),
      SOA_VIEW_VALUE(hitsLayout, clusterSizeX),
      SOA_VIEW_VALUE(hitsLayout, clusterSizeY)
    )
  )
  
  using HitsView = HitsViewTemplate<>;
  
  GENERATE_SOA_LAYOUT(SupportObjectsLayoutTemplate,
    // This is the end of the data which is transferred to host. The following columns are supporting 
    // objects, not transmitted 
    
    // Supporting data (32 bits aligned)
    SOA_COLUMN(TrackingRecHit2DSOAStore::PhiBinner::index_type, phiBinnerStorage),
          
    // global coordinates (not transmitted)
    SOA_COLUMN(int16_t, iphi),
          
    // cluster properties (not transmitted)
    SOA_COLUMN(uint16_t, detectorIndex)
  );
  
  // The support objects layouts also not use default alignment but a more relaxed one.
  using SupportObjectsLayout = SupportObjectsLayoutTemplate<sizeof(TrackingRecHit2DSOAStore::PhiBinner::index_type)>;
  
  GENERATE_SOA_VIEW(HitsAndSupportViewTemplate,
    SOA_VIEW_LAYOUT_LIST(
      SOA_VIEW_LAYOUT(HitsLayout, hitsLayout),
      SOA_VIEW_LAYOUT(SupportObjectsLayout, supportObjectsLayout)
    ),
    SOA_VIEW_VALUE_LIST(
      SOA_VIEW_VALUE(hitsLayout, xLocal),
      SOA_VIEW_VALUE(hitsLayout, yLocal),
      SOA_VIEW_VALUE(hitsLayout, xerrLocal),
      SOA_VIEW_VALUE(hitsLayout, yerrLocal),
      
      SOA_VIEW_VALUE(hitsLayout, xGlobal),
      SOA_VIEW_VALUE(hitsLayout, yGlobal),
      SOA_VIEW_VALUE(hitsLayout, zGlobal),
      SOA_VIEW_VALUE(hitsLayout, rGlobal),
      
      SOA_VIEW_VALUE(hitsLayout, charge),
      SOA_VIEW_VALUE(hitsLayout, clusterSizeX),
      SOA_VIEW_VALUE(hitsLayout, clusterSizeY),
      
      SOA_VIEW_VALUE(supportObjectsLayout, phiBinnerStorage),
      SOA_VIEW_VALUE(supportObjectsLayout, iphi),
      SOA_VIEW_VALUE(supportObjectsLayout, detectorIndex)
    )
  );
  
  using HitsAndSupportView = HitsAndSupportViewTemplate<sizeof(TrackingRecHit2DSOAStore::PhiBinner::index_type)>;
  
  // Shortcut operator saving the explicit calls to view in usage.
  __device__ __forceinline__ HitsAndSupportView::element operator[] (size_t index) {
    return m_hitsAndSupportView[index]; 
  }
  __device__ __forceinline__ HitsAndSupportView::const_element operator[] (size_t index) const {
    return m_hitsAndSupportView[index];
  }
  
  __device__ __forceinline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  __device__ __forceinline__ uint32_t hitsModuleStart(int i) const { return __ldg(m_hitsModuleStart + i); }

  __device__ __forceinline__ uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  __device__ __forceinline__ uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  __device__ __forceinline__ PhiBinner& phiBinner() { return *m_phiBinner; }
  __device__ __forceinline__ PhiBinner const& phiBinner() const { return *m_phiBinner; }

  __device__ __forceinline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  __device__ __forceinline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

private:
  // hits layout
  HitsLayout m_hitsLayout;
  // supporting objects layout
  SupportObjectsLayout m_supportObjectsLayout;
  // Global view simplifying usage
  HitsAndSupportView m_hitsAndSupportView;
  
  // individually defined supporting objects
  // m_averageGeometry is corrected for beam spot, not sure where to host it otherwise
  AverageGeometry* m_averageGeometry;              // owned by TrackingRecHit2DHeterogeneous
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  PhiBinner* m_phiBinner;
  PhiBinner::index_type* m_phiBinnerStorage;

  uint32_t m_nHits;
};

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h