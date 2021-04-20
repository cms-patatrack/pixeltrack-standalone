#ifndef alpakaConfigAcc_h_
#define alpakaConfigAcc_h_

#include "AlpakaCore/alpakaConfigCommon.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
namespace alpaka_cuda_async {
  using namespace alpaka_common;
  using Acc1 = alpaka::AccGpuCudaRt<Dim1, Extent>;
  using Acc2 = alpaka::AccGpuCudaRt<Dim2, Extent>;
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
  using Queue = alpaka::QueueCpuNonBlocking;
}  // namespace alpaka_omp4_async

#endif  // ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp4_async
#endif  // ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using DevAcc1 = alpaka::Dev<Acc1>;
  using DevAcc2 = alpaka::Dev<Acc2>;
  using PltfAcc1 = alpaka::Pltf<DevAcc1>;
  using PltfAcc2 = alpaka::Pltf<DevAcc2>;

  template <class TData>
  using AlpakaAccBuf1 = alpaka::Buf<Acc1, TData, Dim1, Idx>;

  template <class TData>
  using AlpakaAccBuf2 = alpaka::Buf<Acc2, TData, Dim2, Idx>;

  template <typename TData>
  using AlpakaDeviceBuf = AlpakaAccBuf1<TData>;

  template <typename TData>
  using AlpakaDeviceView = alpaka::ViewPlainPtr<DevAcc1, TData, Dim1, Idx>;

  template <typename TData>
  using AlpakaDeviceSubView = alpaka::ViewSubView<DevAcc1, TData, Dim1, Idx>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // alpakaConfigAcc_h_
