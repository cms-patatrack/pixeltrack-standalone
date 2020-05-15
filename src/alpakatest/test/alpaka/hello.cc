#include <iostream>

#include "AlpakaCore/alpakaConfig.h"

int main() {
  std::cout << "Hello from "
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
            << "CPU serial"
#elif defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
            << "CPU TBB"
#elif defined ALPAKA_ACC_GPU_CUDA_ENABLED
            << "CUDA"
#endif
            << " backend" << std::endl;
  return 0;
}
