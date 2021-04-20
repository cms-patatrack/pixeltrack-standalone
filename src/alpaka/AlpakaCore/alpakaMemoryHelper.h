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

    template <typename TData>
      auto createHostView(const DevHost& host, TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<DevHost, TData, Dim1, Idx>(data, host, extent);
    }

    template <typename TData>
      auto createHostView(const DevHost& host, TData* data) {
      return alpaka::ViewPlainPtr<DevHost, TData, Dim1, Idx>(data, host, 1u);
    }

    template <typename TData, typename TDev>
      auto allocDeviceBuf(const TDev& device, const Extent& extent) {
      return alpaka::allocBuf<TData, Idx>(device, extent); 
    }

    template <typename TData, typename TDev>
      auto allocDeviceBuf(const TDev& device) {
      return alpaka::allocBuf<TData, Idx>(device, 1u); 
    }

    template <typename TData, typename TDev>
      auto createDeviceView(const TDev& device, const TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<TDev, TData, Dim1, Idx>(const_cast<TData*>(data), device, extent); // TO DO: Obviously aweful: why no view constructor inside alpaka library with a const TData* argument?
    }

    template <typename TData, typename TDev>
      auto createDeviceView(const TDev& device, const TData* data) {
      return alpaka::ViewPlainPtr<TDev, TData, Dim1, Idx>(const_cast<TData*>(data), device, 1u); // TO DO: Obviously aweful: why no view constructor inside alpaka library with a const TData* argument?
    }

    template <typename TData, typename TDev>
      auto createDeviceView(const TDev& device, TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<TDev, TData, Dim1, Idx>(data, device, extent);
    }

    template <typename TData, typename TDev>
      auto createDeviceView(const TDev& device, TData* data) {
      return alpaka::ViewPlainPtr<TDev, TData, Dim1, Idx>(data, device, 1u);
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


/*
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace cms {
    namespace alpakatools {



      template<typename TData>
	auto sliceOnDevice(Queue& devQueue, AlpakaDeviceBuf<TData> bufferToBeSliced, const Extent& subViewExtents, const Offsets& offsets) -> AlpakaDeviceBuf<TData>
	{
	  AlpakaDeviceBuf<TData> slicedBuffer = ::cms::alpakatools::allocDeviceBuf<TData>(device, subViewExtents);
	  // Create a subView with a possible offset.
	  SubView<TData> subView = SubView<TData>(bufferToBeSliced, subViewExtents, offsets);
	  // Copy the subView into a new buffer.
	  alpaka::memcpy(devQueue, slicedBuffer, subView, subViewExtents);
	  return slicedBuffer;
	}


    }  // namespace alpakatools
  }  // namespace cms
  }*/


#endif  // ALPAKAMEMORYHELPER_H
