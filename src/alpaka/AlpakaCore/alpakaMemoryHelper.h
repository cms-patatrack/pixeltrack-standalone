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

  template <typename TData, typename TQueue>
  auto allocDeviceBuf(TQueue& queue, Extent extent) {
    return alpaka::allocAsyncBuf<TData, Idx>(queue, extent);
  }

  template <typename TData, typename TDevice>
  auto createDeviceView(TDevice const& device, TData const* data, Extent extent) {
    return alpaka::ViewPlainPtr<TDevice, const TData, Dim1D, Idx>(data, device, extent);
  }

  template <typename TData, typename TDevice>
  auto createDeviceView(TDevice const& device, TData* data, Extent extent) {
    return alpaka::ViewPlainPtr<TDevice, TData, Dim1D, Idx>(data, device, extent);
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_alpakaMemoryHelper_h
