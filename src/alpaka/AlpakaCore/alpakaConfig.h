#ifndef AlpakaCore_alpakaConfig_h
#define AlpakaCore_alpakaConfig_h

#include <type_traits>

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

// convert the macro argument to a null-terminated quoted string
#define STRINGIFY_(ARG) #ARG
#define STRINGIFY(ARG) STRINGIFY_(ARG)

// trick to force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification inside DEFINE_FWK_MODULE
#define DEFINE_FWK_ALPAKA_MODULE2(name) DEFINE_FWK_MODULE(name)
#define DEFINE_FWK_ALPAKA_MODULE(name) DEFINE_FWK_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(name) DEFINE_FWK_EVENTSETUP_MODULE(name)
#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(name) \
  DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
namespace alpaka_cuda_async {
  using namespace alpaka_common;
  using Acc1D = alpaka::AccGpuCudaRt<Dim1D, Extent>;
  using Acc2D = alpaka::AccGpuCudaRt<Dim2D, Extent>;
  using Queue = alpaka::QueueCudaRtNonBlocking;

  using Device = alpaka::Dev<Acc1D>;
  using Platform = alpaka::Pltf<Device>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Acc2D>>,
                STRINGIFY(alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>) " and " STRINGIFY(
                    alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>) " are different types.");
  static_assert(std::is_same_v<Platform, alpaka::Pltf<alpaka::Dev<Acc2D>>>,
                STRINGIFY(alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>>) " and " STRINGIFY(
                    alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>>) " are different types.");

  using Event = alpaka::Event<Queue>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Queue>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Queue types.");
  static_assert(std::is_same_v<Device, alpaka::Dev<Event>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Event types.");

}  // namespace alpaka_cuda_async

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cuda
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_cuda_async
#endif  // ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
namespace alpaka_serial_sync {
  using namespace alpaka_common;
  using Acc1D = alpaka::AccCpuSerial<Dim1D, Extent>;
  using Acc2D = alpaka::AccCpuSerial<Dim2D, Extent>;
  using Queue = alpaka::QueueCpuBlocking;

  using Device = alpaka::Dev<Acc1D>;
  using Platform = alpaka::Pltf<Device>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Acc2D>>,
                STRINGIFY(alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>) " and " STRINGIFY(
                    alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>) " are different types.");
  static_assert(std::is_same_v<Platform, alpaka::Pltf<alpaka::Dev<Acc2D>>>,
                STRINGIFY(alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>>) " and " STRINGIFY(
                    alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>>) " are different types.");

  using Event = alpaka::Event<Queue>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Queue>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Queue types.");
  static_assert(std::is_same_v<Device, alpaka::Dev<Event>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Event types.");

}  // namespace alpaka_serial_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_serial_sync
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
namespace alpaka_tbb_async {
  using namespace alpaka_common;
  using Acc1D = alpaka::AccCpuTbbBlocks<Dim1D, Extent>;
  using Acc2D = alpaka::AccCpuTbbBlocks<Dim2D, Extent>;
  using Queue = alpaka::QueueCpuNonBlocking;

  using Device = alpaka::Dev<Acc1D>;
  using Platform = alpaka::Pltf<Device>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Acc2D>>,
                STRINGIFY(alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>) " and " STRINGIFY(
                    alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>) " are different types.");
  static_assert(std::is_same_v<Platform, alpaka::Pltf<alpaka::Dev<Acc2D>>>,
                STRINGIFY(alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>>) " and " STRINGIFY(
                    alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>>) " are different types.");

  using Event = alpaka::Event<Queue>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Queue>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Queue types.");
  static_assert(std::is_same_v<Device, alpaka::Dev<Event>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Event types.");

}  // namespace alpaka_tbb_async

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_tbb_async
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
namespace alpaka_omp2_async {
  using namespace alpaka_common;
  using Acc1D = alpaka::AccCpuOmp2Blocks<Dim1D, Extent>;
  using Acc2D = alpaka::AccCpuOmp2Blocks<Dim2D, Extent>;
  using Queue = alpaka::QueueCpuNonBlocking;

  using Device = alpaka::Dev<Acc1D>;
  using Platform = alpaka::Pltf<Device>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Acc2D>>,
                STRINGIFY(alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>) " and " STRINGIFY(
                    alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>) " are different types.");
  static_assert(std::is_same_v<Platform, alpaka::Pltf<alpaka::Dev<Acc2D>>>,
                STRINGIFY(alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>>) " and " STRINGIFY(
                    alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>>) " are different types.");

  using Event = alpaka::Event<Queue>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Queue>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Queue types.");
  static_assert(std::is_same_v<Device, alpaka::Dev<Event>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Event types.");

}  // namespace alpaka_omp2_async

#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp2_async
#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
namespace alpaka_omp4_async {
  using namespace alpaka_common;
  using Acc1D = alpaka::AccCpuOmp4<Dim1D, Extent>;
  using Acc2D = alpaka::AccCpuOmp4<Dim2D, Extent>;
  using Queue = alpaka::QueueCpuNonBlocking;

  using Device = alpaka::Dev<Acc1D>;
  using Platform = alpaka::Pltf<Device>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Acc2D>>,
                STRINGIFY(alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>) " and " STRINGIFY(
                    alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>) " are different types.");
  static_assert(std::is_same_v<Platform, alpaka::Pltf<alpaka::Dev<Acc2D>>>,
                STRINGIFY(alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>>) " and " STRINGIFY(
                    alpaka::Pltf<alpaka::Dev<::ALPAKA_ACCELERATOR_NAMESPACE::Acc2D>>) " are different types.");

  using Event = alpaka::Event<Queue>;
  static_assert(std::is_same_v<Device, alpaka::Dev<Queue>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Queue types.");
  static_assert(std::is_same_v<Device, alpaka::Dev<Event>>,
                STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE) " has incompatible Accelerator and Event types.");

}  // namespace alpaka_omp4_async

#endif  // ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp4_async
#endif  // ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND

#endif  // AlpakaCore_alpakaConfig_h
