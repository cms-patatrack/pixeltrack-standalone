#ifndef ALPAKAMEMORYHELPER_H
#define ALPAKAMEMORYHELPER_H

#include "AlpakaCore/alpakaConfig.h"

using namespace alpaka_common;

namespace cms {
  namespace alpakatools {

    template <typename TData, typename TDev>
      auto allocHostBuf(const TDev& host, const Extent& extent) {
      return alpaka::allocBuf<TData, Idx>(host, extent); 
    }

    template <typename TData, typename TDev>
      auto allocHostBuf(const TDev& host) {
      return alpaka::allocBuf<TData, Idx>(host, 1u); 
    }

    template <typename T_Data>
      auto createHostView(const DevHost& host, T_Data* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<DevHost, T_Data, Dim1, Idx>(data, host, extent);
    }

    template <typename T_Data>
      auto createHostView(const DevHost& host, T_Data* data) {
      return alpaka::ViewPlainPtr<DevHost, T_Data, Dim1, Idx>(data, host, 1u);
    }

    template <typename TData, typename TDev>
      auto allocDeviceBuf(const TDev& device, const Extent& extent) {
      return alpaka::allocBuf<TData, Idx>(device, extent); 
    }

    template <typename TData, typename TDev>
      auto allocDeviceBuf(const TDev& device) {
      return alpaka::allocBuf<TData, Idx>(device, 1u); 
    }

    template<typename TQueue, typename TViewDst, typename TViewSrc>
      auto memcpy(TQueue& queue, TViewDst& dest, const TViewSrc& src, const Extent& extent) { 
      return alpaka::memcpy(queue, dest, src, extent);
    }

    template<typename TQueue, typename TViewDst, typename TViewSrc>
      auto memcpy(TQueue& queue, TViewDst& dest, const TViewSrc& src) { 
      return alpaka::memcpy(queue, dest, src, 1u);
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAMEMORYHELPER_H
