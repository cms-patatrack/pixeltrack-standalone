#ifndef KokkosCore_kokkosConfig_h
#define KokkosCore_kokkosConfig_h

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_BACKEND_SERIAL
using KokkosExecSpace = Kokkos::Serial;
#define KOKKOS_NAMESPACE kokkos_serial
#elif defined KOKKOS_BACKEND_CUDA
using KokkosExecSpace = Kokkos::Cuda;
#define KOKKOS_NAMESPACE kokkos_cuda
#endif

// trick to force expanding KOKKOS_NAMESPACE before stringification inside DEFINE_FWK_MODULE
#define DEFINE_FWK_KOKKOS_MODULE2(name) DEFINE_FWK_MODULE(name)
#define DEFINE_FWK_KOKKOS_MODULE(name) DEFINE_FWK_KOKKOS_MODULE2(KOKKOS_NAMESPACE::name)

#define DEFINE_FWK_KOKKOS_EVENTSETUP_MODULE2(name) DEFINE_FWK_EVENTSETUP_MODULE(name)
#define DEFINE_FWK_KOKKOS_EVENTSETUP_MODULE(name) DEFINE_FWK_KOKKOS_EVENTSETUP_MODULE2(KOKKOS_NAMESPACE::name)

#endif
