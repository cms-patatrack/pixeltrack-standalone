#include <iostream>

#include "KokkosCore/kokkosConfig.h"

int main() {
  std::cout << "Hello from "
#ifdef KOKKOS_BACKEND_SERIAL
            << "CPU serial"
#elif defined KOKKOS_BACKEND_CUDA
            << "CUDA"
#endif
            << " backend" << std::endl;
  return 0;
}
