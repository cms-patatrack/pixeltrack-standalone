#ifndef ALPAKAMEMORYHELPER_H
#define ALPAKAMEMORYHELPER_H

#include "AlpakaCore/alpakaConfig.h"

using namespace alpaka_common;

namespace cms {
  namespace alpakatools {

    template <typename T_Data>
      auto createHostView(const DevHost& host, T_Data* data) {
      return alpaka::ViewPlainPtr<DevHost, T_Data, Dim1, Idx>(data, host, 1u);
    }

    template <typename TData, typename TDev>
      auto allocDeviceBuf(const TDev& device) {
      return alpaka::allocBuf<TData, Idx>(device, 1u); 
    }

    template<typename TQueue, typename TViewDst, typename TViewSrc>
      auto memcpy(TQueue& queue, TViewDst& dest, const TViewSrc& src) { 
      return alpaka::memcpy(queue, dest, src, 1u);
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAMEMORYHELPER_H
