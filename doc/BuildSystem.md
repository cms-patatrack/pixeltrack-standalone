# Build system

## Overall description

Work in progress.

## Test program specific notes

### Kokkos

When Kokkos runtime library is built with CUDA backend, all files
including any Kokkos header need to be compiled with `nvcc` (to
leading order, at least). In addition, the aim is to build all the
portable code for both CPU serial and CUDA backends. There is,
however, some backend-independent code that needs to be built with
`nvcc` because of the inclusion of a Kokkos header. The [build
rules](../src/kokkostest/Makefile) to handle all that are as follows
* Code including a Kokkos header that is backend-independent should be placed in `<package>/kokkoshost/<filename>.cc`
  * Such files are compiled with `nvcc`, but no special macros are set
  * The resulting object file participates in the device link for that package
* Portable (i.e. backend-dependent) code should be placed in `<package>/kokkos/<filename>.cc`
  * Such files are compiled with `nvcc` twice, once for CPU serial backend with `KOKKOS_BACKEND_SERIAL` macro defined, and once for the CUDA backend with `KOKKOS_BACKEND_CUDA` macro defined
    * We should try hard to avoid depending on these macros in user code, and try to provide abstractions in [`KokkosCore/kokkosConfig.h`](../src/kokkostest/KokkosCore/kokkosConfig.h) instead
  * The resulting object files participate in the device link for that package
