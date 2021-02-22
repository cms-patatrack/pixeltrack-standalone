#ifndef alpakaConfig_h_
#define alpakaConfig_h_

#include <alpaka/alpaka.hpp>

namespace alpaka_common {
  using Idx = uint32_t;
  using Extent = uint32_t;
  using DevHost = alpaka::DevCpu;
  using PltfHost = alpaka::Pltf<DevHost>;

  using Dim1 = alpaka::DimInt<1u>;
  using Dim2 = alpaka::DimInt<2u>;

  template <typename T_Dim>
  using Vec = alpaka::Vec<T_Dim, Idx>;
  using Vec1 = Vec<Dim1>;
  using Vec2 = Vec<Dim2>;

  template <typename T_Dim>
  using WorkDiv = alpaka::WorkDivMembers<T_Dim, Idx>;
  using WorkDiv1 = WorkDiv<Dim1>;
  using WorkDiv2 = WorkDiv<Dim2>;
}  // namespace alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
namespace alpaka_cuda_async {
  using namespace alpaka_common;
  using Acc1 = alpaka::AccGpuCudaRt<Dim1, Extent>;
  using Acc2 = alpaka::AccGpuCudaRt<Dim2, Extent>;
  using DevAcc1 = alpaka::Dev<Acc1>;
  using DevAcc2 = alpaka::Dev<Acc2>;
  using PltfAcc1 = alpaka::Pltf<DevAcc1>;
  using PltfAcc2 = alpaka::Pltf<DevAcc2>;

  template <class T_Data>
  using AlpakaAccBuf2 = alpaka::Buf<Acc2, T_Data, Dim2, Idx>;

  using Queue = alpaka::QueueCudaRtNonBlocking;
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
  using Acc1 = alpaka::AccCpuSerial<Dim1, Extent>;
  using Acc2 = alpaka::AccCpuSerial<Dim2, Extent>;
  using DevAcc1 = alpaka::Dev<Acc1>;
  using DevAcc2 = alpaka::Dev<Acc2>;
  using PltfAcc1 = alpaka::Pltf<DevAcc1>;
  using PltfAcc2 = alpaka::Pltf<DevAcc2>;

  template <class T_Data>
  using AlpakaAccBuf2 = alpaka::Buf<Acc2, T_Data, Dim2, Idx>;

  using Queue = alpaka::QueueCpuBlocking;
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
  using Acc1 = alpaka::AccCpuTbbBlocks<Dim1, Extent>;
  using Acc2 = alpaka::AccCpuTbbBlocks<Dim2, Extent>;
  using DevAcc1 = alpaka::Dev<Acc1>;
  using DevAcc2 = alpaka::Dev<Acc2>;
  using PltfAcc1 = alpaka::Pltf<DevAcc1>;
  using PltfAcc2 = alpaka::Pltf<DevAcc2>;

  template <class T_Data>
  using AlpakaAccBuf2 = alpaka::Buf<Acc2, T_Data, Dim2, Idx>;

  using Queue = alpaka::QueueCpuNonBlocking;
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
  using Acc1 = alpaka::AccCpuOmp2Blocks<Dim1, Extent>;
  using Acc2 = alpaka::AccCpuOmp2Blocks<Dim2, Extent>;
  using DevAcc1 = alpaka::Dev<Acc1>;
  using DevAcc2 = alpaka::Dev<Acc2>;
  using PltfAcc1 = alpaka::Pltf<DevAcc1>;
  using PltfAcc2 = alpaka::Pltf<DevAcc2>;

  template <class T_Data>
  using AlpakaAccBuf2 = alpaka::Buf<Acc2, T_Data, Dim2, Idx>;

  using Queue = alpaka::QueueCpuNonBlocking;
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
  using Acc1 = alpaka::AccCpuOmp4<Dim1, Extent>;
  using Acc2 = alpaka::AccCpuOmp4<Dim2, Extent>;
  using DevAcc1 = alpaka::Dev<Acc1>;
  using DevAcc2 = alpaka::Dev<Acc2>;
  using PltfAcc1 = alpaka::Pltf<DevAcc1>;
  using PltfAcc2 = alpaka::Pltf<DevAcc2>;

  template <class T_Data>
  using AlpakaAccBuf2 = alpaka::Buf<Acc2, T_Data, Dim2, Idx>;

  using Queue = alpaka::QueueCpuNonBlocking;
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
