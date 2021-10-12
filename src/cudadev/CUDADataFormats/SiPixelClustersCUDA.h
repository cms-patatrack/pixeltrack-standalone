#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/cudaCompat.h"
#include "DataFormats/SoAStore.h"

#include <cuda_runtime.h>

class SiPixelClustersCUDA {
public:
  generate_SoA_store(DeviceStore,
    SoA_column(uint32_t, moduleStart),  // index of the first pixel of each module
    SoA_column(uint32_t, clusInModule), // number of clusters found in each module
    SoA_column(uint32_t, moduleId),     // module id of each module

    // originally from rechits
    SoA_column(uint32_t, clusModuleStart) // index of the first cluster of each module
  );

  explicit SiPixelClustersCUDA();
  explicit SiPixelClustersCUDA(size_t maxModules, cudaStream_t stream);
  ~SiPixelClustersCUDA() = default;

  SiPixelClustersCUDA(const SiPixelClustersCUDA &) = delete;
  SiPixelClustersCUDA &operator=(const SiPixelClustersCUDA &) = delete;
  SiPixelClustersCUDA(SiPixelClustersCUDA &&) = default;
  SiPixelClustersCUDA &operator=(SiPixelClustersCUDA &&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

  uint32_t nClusters() const { return nClusters_h; }

  uint32_t *moduleStart() { return deviceStore_.moduleStart(); }
  uint32_t *clusInModule() { return deviceStore_.clusInModule(); }
  uint32_t *moduleId() { return deviceStore_.moduleId(); }
  uint32_t *clusModuleStart() { return deviceStore_.clusModuleStart(); }

  uint32_t const *moduleStart() const { return deviceStore_.moduleStart(); }
  uint32_t const *clusInModule() const { return deviceStore_.clusInModule(); }
  uint32_t const *moduleId() const { return deviceStore_.moduleId(); }
  uint32_t const *clusModuleStart() const { return deviceStore_.clusModuleStart(); }

  const DeviceStore store() const { return deviceStore_; }

private:
  cms::cuda::device::unique_ptr<std::byte[]> data_d;         // Single SoA storage
  DeviceStore deviceStore_;
  uint32_t nClusters_h = 0;
};

#endif  // CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
