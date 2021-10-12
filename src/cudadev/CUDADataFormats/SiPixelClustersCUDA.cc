#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(): data_d(), deviceStore_(data_d.get(), 0) {}

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxModules, cudaStream_t stream)
    : data_d(cms::cuda::make_device_unique<std::byte[]>(DeviceStore::computeDataSize(maxModules), stream)),
      deviceStore_(data_d.get(), maxModules)
{}
