#ifndef KokkosCore_kokkosConfig_h
#define KokkosCore_kokkosConfig_h

#include <Kokkos_Core.hpp>
#include "KokkosCore/kokkosConfigCommon.h"

#ifdef KOKKOS_BACKEND_SERIAL
using KokkosExecSpace = Kokkos::Serial;
#define KOKKOS_NAMESPACE kokkos_serial
#elif defined KOKKOS_BACKEND_CUDA
using KokkosExecSpace = Kokkos::Cuda;
#define KOKKOS_NAMESPACE kokkos_cuda
#endif

// this is meant mostly for unit tests
template <typename ExecSpace>
struct KokkosBackend;
template <>
struct KokkosBackend<Kokkos::Serial> {
  static constexpr auto value = kokkos_common::InitializeScopeGuard::Backend::SERIAL;
};
template <>
struct KokkosBackend<Kokkos::Cuda> {
  static constexpr auto value = kokkos_common::InitializeScopeGuard::Backend::CUDA;
};

// trick to force expanding KOKKOS_NAMESPACE before stringification inside DEFINE_FWK_MODULE
#define DEFINE_FWK_KOKKOS_MODULE2(name) DEFINE_FWK_MODULE(name)
#define DEFINE_FWK_KOKKOS_MODULE(name) DEFINE_FWK_KOKKOS_MODULE2(KOKKOS_NAMESPACE::name)

#endif
