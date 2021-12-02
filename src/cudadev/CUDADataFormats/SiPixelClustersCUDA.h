#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/cudaCompat.h"
#include "DataFormats/SoAStore.h"
#include "DataFormats/SoAView.h"

#include <cuda_runtime.h>

class SiPixelClustersCUDA {
public:
  generate_SoA_store(DeviceLayoutTemplate,
    SoA_column(uint32_t, moduleStart),  // index of the first pixel of each module
    SoA_column(uint32_t, clusInModule), // number of clusters found in each module
    SoA_column(uint32_t, moduleId),     // module id of each module

    // originally from rechits
    SoA_column(uint32_t, clusModuleStart) // index of the first cluster of each module
  );
  
  // We use all defaults for the template parameters.
  using DeviceLayout = DeviceLayoutTemplate<>;

    generate_SoA_view(DeviceViewTemplate,
    SoA_view_store_list(SoA_view_store(DeviceLayout, deviceLayout)),
    SoA_view_value_list(
      SoA_view_value(deviceLayout, moduleStart),  // index of the first pixel of each module
      SoA_view_value(deviceLayout, clusInModule), // number of clusters found in each module
      SoA_view_value(deviceLayout, moduleId),     // module id of each module
  
      // originally from rechits
      SoA_view_value(deviceLayout, clusModuleStart) // index of the first cluster of each module
    )
  );
  
  using DeviceView = DeviceViewTemplate<>;
  
  generate_SoA_const_view(DeviceConstViewTemplate,
    SoA_view_store_list(SoA_view_store(DeviceView, deviceView)),
    SoA_view_value_list(
      SoA_view_value(deviceView, moduleStart),  // index of the first pixel of each module
      SoA_view_value(deviceView, clusInModule), // number of clusters found in each module
      SoA_view_value(deviceView, moduleId),     // module id of each module
  
      // originally from rechits
      SoA_view_value(deviceView, clusModuleStart) // index of the first cluster of each module
    )
  );
  
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
