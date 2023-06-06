#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"

SiPixelClustersCUDA::SiPixelClustersCUDA() : data_d(), deviceLayout_(data_d.get(), 0), deviceView_(deviceLayout_) {}

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxModules, cudaStream_t stream)
    : data_d(cms::cuda::make_device_unique<std::byte[]>(DeviceLayout::computeDataSize(maxModules), stream)),
      deviceLayout_(data_d.get(), maxModules),
      deviceView_(deviceLayout_) {}
