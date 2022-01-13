#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/cudaCompat.h"
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

#include <cuda_runtime.h>

class SiPixelClustersCUDA {
public:
  GENERATE_SOA_LAYOUT(DeviceLayoutTemplate,
    SOA_COLUMN(uint32_t, moduleStart),  // index of the first pixel of each module
    SOA_COLUMN(uint32_t, clusInModule), // number of clusters found in each module
    SOA_COLUMN(uint32_t, moduleId),     // module id of each module

    // originally from rechits
    SOA_COLUMN(uint32_t, clusModuleStart) // index of the first cluster of each module
  )
  
  // We use all defaults for the template parameters.
  using DeviceLayout = DeviceLayoutTemplate<>;

    GENERATE_SOA_VIEW(DeviceViewTemplate,
    SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(DeviceLayout, deviceLayout)),
    SOA_VIEW_VALUE_LIST(
      SOA_VIEW_VALUE(deviceLayout, moduleStart),  // index of the first pixel of each module
      SOA_VIEW_VALUE(deviceLayout, clusInModule), // number of clusters found in each module
      SOA_VIEW_VALUE(deviceLayout, moduleId),     // module id of each module
  
      // originally from rechits
      SOA_VIEW_VALUE(deviceLayout, clusModuleStart) // index of the first cluster of each module
    )
  )
  
  using DeviceView = DeviceViewTemplate<>;
  
  GENERATE_SOA_CONST_VIEW(DeviceConstViewTemplate,
    SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(DeviceView, deviceView)),
    SOA_VIEW_VALUE_LIST(
      SOA_VIEW_VALUE(deviceView, moduleStart),  // index of the first pixel of each module
      SOA_VIEW_VALUE(deviceView, clusInModule), // number of clusters found in each module
      SOA_VIEW_VALUE(deviceView, moduleId),     // module id of each module
  
      // originally from rechits
      SOA_VIEW_VALUE(deviceView, clusModuleStart) // index of the first cluster of each module
    )
  )
  
  using DeviceConstView = DeviceConstViewTemplate<>;
  
  explicit SiPixelClustersCUDA();
  explicit SiPixelClustersCUDA(size_t maxModules, cudaStream_t stream);
  ~SiPixelClustersCUDA() = default;

  SiPixelClustersCUDA(const SiPixelClustersCUDA &) = delete;
  SiPixelClustersCUDA &operator=(const SiPixelClustersCUDA &) = delete;
  SiPixelClustersCUDA(SiPixelClustersCUDA &&) = default;
  SiPixelClustersCUDA &operator=(SiPixelClustersCUDA &&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

  uint32_t nClusters() const { return nClusters_h; }

  uint32_t *moduleStart() { return deviceView_.moduleStart(); }
  uint32_t *clusInModule() { return deviceView_.clusInModule(); }
  uint32_t *moduleId() { return deviceView_.moduleId(); }
  uint32_t *clusModuleStart() { return deviceView_.clusModuleStart(); }

  uint32_t const *moduleStart() const { return deviceView_.moduleStart(); }
  uint32_t const *clusInModule() const { return deviceView_.clusInModule(); }
  uint32_t const *moduleId() const { return deviceView_.moduleId(); }
  uint32_t const *clusModuleStart() const { return deviceView_.clusModuleStart(); }

  DeviceConstView view() const { return DeviceConstView(deviceView_); }

private:
  cms::cuda::device::unique_ptr<std::byte[]> data_d;         // Single SoA storage
  DeviceLayout deviceLayout_;
  DeviceView deviceView_;
  
  uint32_t nClusters_h = 0;
};

#endif  // CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
