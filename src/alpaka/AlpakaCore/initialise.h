#ifndef INITIALIZE_H
#define INITIALIZE_H

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SUPPORTED
namespace alpaka_serial_sync {
  void initialise();
}
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SUPPORTED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_SUPPORTED
namespace alpaka_tbb_async {
  void initialise();
}
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_SUPPORTED

#ifdef ALPAKA_ACC_GPU_CUDA_SUPPORTED
namespace alpaka_cuda_async {
  void initialise();
}
#endif  // ALPAKA_ACC_GPU_CUDA_SUPPORTED

void initialise();

#endif  // INITIALIZE_H
