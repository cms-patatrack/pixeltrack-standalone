#ifndef alpakaConfigHost_h_
#define alpakaConfigHost_h_

#include <alpaka/alpaka.hpp>

namespace alpaka_common {
  using Idx = uint32_t;
  using Extent = uint32_t;
  using Offsets = Extent;
  using DevHost = alpaka::DevCpu;
  using PltfHost = alpaka::Pltf<DevHost>;

  using Dim1 = alpaka::DimInt<1u>;
  using Dim2 = alpaka::DimInt<2u>;

  template <typename TDim>
  using Vec = alpaka::Vec<TDim, Idx>;
  using Vec1 = Vec<Dim1>;
  using Vec2 = Vec<Dim2>;

  template <typename TDim>
  using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
  using WorkDiv1 = WorkDiv<Dim1>;
  using WorkDiv2 = WorkDiv<Dim2>;

  template <typename TData>
    using AlpakaHostBuf = alpaka::Buf<DevHost, TData, Dim1, Idx>;

  template <typename TData>
    using AlpakaHostView = alpaka::ViewPlainPtr<DevHost, TData, Dim1, Idx>;
}  // namespace alpaka_common


// trick to force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification inside DEFINE_FWK_MODULE
#define DEFINE_FWK_ALPAKA_MODULE2(name) DEFINE_FWK_MODULE(name)
#define DEFINE_FWK_ALPAKA_MODULE(name) DEFINE_FWK_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(name) DEFINE_FWK_EVENTSETUP_MODULE(name)
#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(name) DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#endif  // alpakaConfigHost_h_
