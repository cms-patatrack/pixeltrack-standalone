#ifndef alpakaConfigAcc_h_
#define alpakaConfigAcc_h_

#include "AlpakaCore/alpakaConfigCommon.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
namespace alpaka_cuda_async {
  using namespace alpaka_common;
  using Acc1D = alpaka::AccGpuCudaRt<Dim1D, Extent>;
  using Acc2D = alpaka::AccGpuCudaRt<Dim2D, Extent>;
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
  using Acc1D = alpaka::AccCpuSerial<Dim1D, Extent>;
  using Acc2D = alpaka::AccCpuSerial<Dim2D, Extent>;
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
  using Acc1D = alpaka::AccCpuTbbBlocks<Dim1D, Extent>;
  using Acc2D = alpaka::AccCpuTbbBlocks<Dim2D, Extent>;
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
  using Acc1D = alpaka::AccCpuOmp2Blocks<Dim1D, Extent>;
  using Acc2D = alpaka::AccCpuOmp2Blocks<Dim2D, Extent>;
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
  using Acc1D = alpaka::AccCpuOmp4<Dim1D, Extent>;
  using Acc2D = alpaka::AccCpuOmp4<Dim2D, Extent>;
  using Queue = alpaka::QueueCpuNonBlocking;
}  // namespace alpaka_omp4_async

#endif  // ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp4_async
#endif  // ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using Device = alpaka::Dev<Acc1D>;      // these are independent from the dimensionality
  using Platform = alpaka::Pltf<Device>;  //

  template <class TData>
  using AlpakaAccBuf1D = alpaka::Buf<Acc1D, TData, Dim1D, Idx>;

  template <class TData>
  using AlpakaAccBuf2D = alpaka::Buf<Acc2D, TData, Dim2D, Idx>;

  template <typename TData>
  using AlpakaDeviceBuf = AlpakaAccBuf1D<TData>;

  template <typename TData>
  using AlpakaDeviceView = alpaka::ViewPlainPtr<Device, TData, Dim1D, Idx>;

  template <typename TData>
  using AlpakaDeviceSubView = alpaka::ViewSubView<Device, TData, Dim1D, Idx>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // alpakaConfigAcc_h_
