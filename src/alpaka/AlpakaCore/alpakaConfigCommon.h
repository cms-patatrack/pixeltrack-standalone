#ifndef alpakaConfigHost_h_
#define alpakaConfigHost_h_

#include <alpaka/alpaka.hpp>

namespace alpaka_common {

  // common types and dimensions
  using Idx = uint32_t;
  using Extent = uint32_t;
  using Offsets = Extent;

  using Dim0D = alpaka::DimInt<0u>;
  using Dim1D = alpaka::DimInt<1u>;
  using Dim2D = alpaka::DimInt<2u>;
  using Dim3D = alpaka::DimInt<3u>;

  template <typename TDim>
  using Vec = alpaka::Vec<TDim, Idx>;
  using Vec1D = Vec<Dim1D>;
  using Vec2D = Vec<Dim2D>;
  using Vec3D = Vec<Dim3D>;
  using Scalar = Vec<Dim0D>;

  template <typename TDim>
  using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
  using WorkDiv1D = WorkDiv<Dim1D>;
  using WorkDiv2D = WorkDiv<Dim2D>;
  using WorkDiv3D = WorkDiv<Dim3D>;

  // host types
  using DevHost = alpaka::DevCpu;
  using PltfHost = alpaka::Pltf<DevHost>;

}  // namespace alpaka_common

// trick to force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification inside DEFINE_FWK_MODULE
#define DEFINE_FWK_ALPAKA_MODULE2(name) DEFINE_FWK_MODULE(name)
#define DEFINE_FWK_ALPAKA_MODULE(name) DEFINE_FWK_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(name) DEFINE_FWK_EVENTSETUP_MODULE(name)
#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(name) \
  DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#endif  // alpakaConfigHost_h_
