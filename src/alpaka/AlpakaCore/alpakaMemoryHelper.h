#ifndef AlpakaCore_alpakaMemoryHelper_h
#define AlpakaCore_alpakaMemoryHelper_h

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaDevices.h"

namespace cms::alpakatools {

  // for Extent, Dim1D, Idx
  using namespace alpaka_common;

  template <typename TData>
  auto allocHostBuf(Extent extent) {
    return alpaka::allocBuf<TData, Idx>(host, extent);
  }

  template <typename TData>
  auto createHostView(TData* data, Extent extent) {
    return alpaka::ViewPlainPtr<DevHost, TData, Dim1D, Idx>(data, host, extent);
  }

  template <typename TData, typename TDevice>
  auto allocDeviceBuf(TDevice const& device, Extent extent) {
    return alpaka::allocBuf<TData, Idx>(device, extent);
  }

  template <typename TData, typename TDevice>
  auto createDeviceView(TDevice const& device, TData const* data, Extent extent) {
    return alpaka::ViewPlainPtr<TDevice, const TData, Dim1D, Idx>(data, device, extent);
  }

  template <typename TData, typename TDevice>
  auto createDeviceView(TDevice const& device, TData* data, Extent extent) {
    return alpaka::ViewPlainPtr<TDevice, TData, Dim1D, Idx>(data, device, extent);
  }

  template <typename TData>
  inline size_t nbytesFromExtent(const Extent& extent) {
    return (sizeof(TData) * extent);
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_alpakaMemoryHelper_h
