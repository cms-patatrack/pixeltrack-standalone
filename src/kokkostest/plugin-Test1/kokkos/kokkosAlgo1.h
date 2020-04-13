#ifndef Test1_kokkosAlgo1_h
#define Test1_kokkosAlgo1_h

#include "KokkosCore/kokkosConfig.h"

namespace KOKKOS_NAMESPACE {
  Kokkos::View<float*, KokkosExecSpace> kokkosAlgo1();
}

#endif
