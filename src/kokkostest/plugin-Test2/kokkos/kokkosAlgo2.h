#ifndef Test2_kokkosAlgo2_h
#define Test2_kokkosAlgo2_h

#include "KokkosCore/kokkosConfig.h"

namespace KOKKOS_NAMESPACE {
  Kokkos::View<float*, KokkosExecSpace> kokkosAlgo2();
}

#endif
