#ifndef alpakaConfig_h_
#define alpakaConfig_h_

#include <alpaka/alpaka.hpp>

namespace alpaka_common {
  using Dim = alpaka::dim::DimInt<1u>;
  using Dim2 = alpaka::dim::DimInt<2u>;
  using Idx = uint32_t;
  using Extent = uint32_t;
  using DevHost = alpaka::dev::DevCpu;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
  using WorkDiv2 = alpaka::workdiv::WorkDivMembers<Dim2, Idx>;
  using Vec = alpaka::vec::Vec<Dim, Idx>;
  using Vec2 = alpaka::vec::Vec<Dim2, Idx>;
}  // namespace alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
namespace alpaka_cuda_async {
  using namespace alpaka_common;
  using Acc = alpaka::acc::AccGpuCudaRt<Dim, Extent>;
  using Acc2 = alpaka::acc::AccGpuCudaRt<Dim2, Extent>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using DevAcc2 = alpaka::dev::Dev<Acc2>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using PltfAcc2 = alpaka::pltf::Pltf<DevAcc2>;
  using Queue = alpaka::queue::QueueCudaRtNonBlocking;
}  // namespace alpaka_cuda_async

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cuda
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_cuda_async
#endif  // ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
namespace alpaka_serial_sync {
  using namespace alpaka_common;
  using Acc = alpaka::acc::AccCpuSerial<Dim, Extent>;
  using Acc2 = alpaka::acc::AccCpuSerial<Dim2, Extent>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using DevAcc2 = alpaka::dev::Dev<Acc2>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using PltfAcc2 = alpaka::pltf::Pltf<DevAcc2>;
  using Queue = alpaka::queue::QueueCpuBlocking;
}  // namespace alpaka_serial_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_serial_sync
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
namespace alpaka_tbb_async {
  using namespace alpaka_common;
  using Acc = alpaka::acc::AccCpuTbbBlocks<Dim, Extent>;
  using Acc2 = alpaka::acc::AccCpuTbbBlocks<Dim2, Extent>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using DevAcc2 = alpaka::dev::Dev<Acc2>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using PltfAcc2 = alpaka::pltf::Pltf<DevAcc2>;
  using Queue = alpaka::queue::QueueCpuNonBlocking;
}  // namespace alpaka_tbb_async

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_tbb_async
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND
namespace alpaka_omp2_async {
  using namespace alpaka_common;
  using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Extent>;
  using Acc2 = alpaka::acc::AccCpuOmp2Blocks<Dim2, Extent>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using DevAcc2 = alpaka::dev::Dev<Acc2>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using PltfAcc2 = alpaka::pltf::Pltf<DevAcc2>;
  using Queue = alpaka::queue::QueueCpuNonBlocking;
}  // namespace alpaka_omp2_async

#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp2_async
#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
#define ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND
namespace alpaka_omp4_async {
  using namespace alpaka_common;
  using Acc = alpaka::acc::AccCpuOmp4<Dim, Extent>;
  using Acc2 = alpaka::acc::AccCpuOmp4<Dim2, Extent>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using DevAcc2 = alpaka::dev::Dev<Acc2>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using PltfAcc2 = alpaka::pltf::Pltf<DevAcc2>;
  using Queue = alpaka::queue::QueueCpuNonBlocking;
}  // namespace alpaka_omp4_async

#endif  // ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp4_async
#endif  // ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND

// trick to force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification inside DEFINE_FWK_MODULE
#define DEFINE_FWK_ALPAKA_MODULE2(name) DEFINE_FWK_MODULE(name)
#define DEFINE_FWK_ALPAKA_MODULE(name) DEFINE_FWK_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(name) DEFINE_FWK_EVENTSETUP_MODULE(name)
#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(name) \
  DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#endif  // alpakaConfig_h_
