#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelROCsStatusAndMappingWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelROCsStatusAndMappingWrapper_h

#include "CUDACore/ESProduct.h"
#include "CUDACore/HostAllocator.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/device_unique_ptr.h"
#include "CondFormats/SiPixelROCsStatusAndMapping.h"

#include <cuda_runtime.h>

#include <set>

class SiPixelROCsStatusAndMappingWrapper {
public:
  /* This is using a layout as the size is needed. TODO: use views when views start embedding size. */
  explicit SiPixelROCsStatusAndMappingWrapper(SiPixelROCsStatusAndMapping const &cablingMap,
                                          std::vector<unsigned char> modToUnp);

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  SiPixelROCsStatusAndMappingConstView getGPUProductAsync(cudaStream_t cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(cudaStream_t cudaStream) const;

private:
  std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> modToUnpDefault;
  bool hasQuality_;

  cms::cuda::host::unique_ptr<SiPixelROCsStatusAndMapping> cablingMapHost;  // host pined memory for cabling map.

  struct GPUData {
    void allocate(cudaStream_t stream) {
      cablingMapDevice = cms::cuda::make_device_unique<SiPixelROCsStatusAndMapping>(stream);
      // Populate the view with individual column pointers
      auto & cmd = *cablingMapDevice;
      cablingMapDeviceView = SiPixelROCsStatusAndMappingConstView(
        pixelgpudetails::MAX_SIZE,
        cmd.fed, // Those are array pointers (in device, but we won't dereference them here).
        cmd.link,
        cmd.roc,
        cmd.rawId,
        cmd.rocInDet,
        cmd.moduleId,
        cmd.badRocs,
        &cmd.size // This is a scalar, we need the address-of operator
      );
    }
    cms::cuda::device::unique_ptr<SiPixelROCsStatusAndMapping> cablingMapDevice;
    SiPixelROCsStatusAndMappingConstView cablingMapDeviceView; // map struct in GPU
    
  };
  cms::cuda::ESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    cms::cuda::device::unique_ptr<unsigned char []> modToUnpDefault;  // pointer to GPU
  };
  cms::cuda::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif
