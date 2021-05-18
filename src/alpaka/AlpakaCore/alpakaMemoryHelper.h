#ifndef ALPAKAMEMORYHELPER_H
#define ALPAKAMEMORYHELPER_H

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaDevices.h"

using namespace alpaka_common;

namespace cms {
  namespace alpakatools {

    template <typename TData>
    auto allocHostBuf(const Extent& extent) {
      return alpaka::allocBuf<TData, Idx>(host, extent);
    }

    template <typename TData>
    auto createHostView(TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<DevHost, TData, Dim1, Idx>(data, host, extent);
    }

    template <typename TData>
    auto allocDeviceBuf(const Extent& extent) {
      return alpaka::allocBuf<TData, Idx>(ALPAKA_ACCELERATOR_NAMESPACE::device, extent);
    }

    template <typename TData>
    auto createDeviceView(const TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1, const TData, Dim1, Idx>(
          data, ALPAKA_ACCELERATOR_NAMESPACE::device, extent);
    }

    template <typename TData>
    auto createDeviceView(TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1, TData, Dim1, Idx>(
          data, ALPAKA_ACCELERATOR_NAMESPACE::device, extent);
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAMEMORYHELPER_H
