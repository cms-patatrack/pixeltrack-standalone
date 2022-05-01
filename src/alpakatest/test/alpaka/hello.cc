#include <iostream>

int main() {
  std::cout << "Hello from the "
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
            << "CPU serial "
#endif
#if defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
            << "CPU TBB "
#endif
#if defined ALPAKA_ACC_GPU_CUDA_ENABLED
            << "CUDA "
#endif
#if defined ALPAKA_ACC_GPU_HIP_ENABLED
            << "HIP/ROCm "
#endif
            << "backend" << std::endl;
  return 0;
}
