# Standalone [Patatrack](https://patatrack.web.cern.ch/patatrack/wiki/) pixel tracking

## Table of contents

* [Introduction](#introduction)
* [Status](#status)
* [Quick recipe](#quick-recipe)
  * [Additional make targets](#additional-make-targets)
  * [Test program specific notes (if any)](#test-program-specific-notes-if-any)
    * [`fwtest`](#fwtest)
    * [`serial`](#serial)
    * [`cudatest`](#cudatest)
    * [`cuda`](#cuda)
    * [`cudadev`](#cudadev)
    * [`cudauvm`](#cudauvm)
    * [`cudacompat`](#cudacompat)
    * [`hip` and `hiptest`](#hip-and-hiptest)
    * [`kokkos` and `kokkostest`](#kokkos-and-kokkostest)
* [Code structure](#code-structure)
* [Build system](#build-system)
* [Contribution guide](#contribution-guide)

## Introduction

The purpose of this package is to explore various performance
portability solutions with the
[Patatrack](https://patatrack.web.cern.ch/patatrack/wiki/) pixel
tracking application. The version here corresponds to
[CMSSW_11_2_0_pre8_Patatrack](https://github.com/cms-patatrack/cmssw/tree/CMSSW_11_2_0_pre8_Patatrack).

The application is designed to require minimal dependencies on the system. All programs require
* GNU Make, `curl`, `md5sum`, `tar`
* C++17 capable compiler. For programs using CUDA that must work with `nvcc`, this means GCC 8, 9, 10 or 11 (since CUDA 11.4.1).
  * testing is currently done with GCC 8
  * not that due to a bug in GCC, GCC 10.3 is not supported

In addition, the individual programs assume the following be found from the system

| Application  | CMake (>= 3.16)    | CUDA 11.2                   | ROCm 4.0               | [Intel oneAPI Base Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html) |
|--------------|--------------------|-----------------------------|------------------------|------------------------------------------------------------------------------------------------------------------|
| `cudatest`   |                    | :heavy_check_mark:          |                        |                                                                                                                  |
| `cuda`       |                    | :heavy_check_mark:          |                        |                                                                                                                  |
| `cudadev`    |                    | :heavy_check_mark:          |                        |                                                                                                                  |
| `cudauvm`    |                    | :heavy_check_mark:          |                        |                                                                                                                  |
| `cudacompat` |                    | :heavy_check_mark:          |                        |                                                                                                                  |
| `hiptest`    |                    |                             | :heavy_check_mark:     |                                                                                                                  |
| `hip`        |                    |                             | :heavy_check_mark:     |                                                                                                                  |
| `kokkostest` | :heavy_check_mark: | :white_check_mark: (1)      | :white_check_mark: (2) |                                                                                                                  |
| `kokkos`     | :heavy_check_mark: | :white_check_mark: (1)      | :white_check_mark: (2) |                                                                                                                  |
| `alpakatest` |                    | :heavy_check_mark:          |                        |                                                                                                                  |
| `alpaka`     |                    | :heavy_check_mark:          |                        |                                                                                                                  |
| `sycltest`   |                    |                             |                        | :heavy_check_mark:                                                                                               |
|              |                    |                             |                        |                                                                                                                  |

1. `kokkos` and `kokkostest` have an optional dependence on CUDA, by default it is required (see [`kokkos` and `kokkostest`](#kokkos-and-kokkostest) for more details)
2. `kokkos` and `kokkostest` have an optional dependence on ROCm, by default it is not required (see [`kokkos` and `kokkostest`](#kokkos-and-kokkostest) for more details)


All other dependencies (listed below) are downloaded and built automatically


| Application  | [TBB](https://github.com/intel/tbb) | [Eigen](http://eigen.tuxfamily.org/) | [Kokkos](https://github.com/kokkos/kokkos) | [Boost](https://www.boost.org/) (1) | [Alpaka](https://github.com/alpaka-group/alpaka) | [libbacktrace](https://github.com/ianlancetaylor/libbacktrace) | [hwloc](https://www.open-mpi.org/projects/hwloc/) |
|--------------|-------------------------------------|--------------------------------------|--------------------------------------------|-------------------------------------|--------------------------------------------------|----------------------------------------------------------------|---------------------------------------------------|
| `fwtest`     | :heavy_check_mark:                  |                                      |                                            |                                     |                                                  |                                                                |                                                   |
| `serial`     | :heavy_check_mark:                  | :heavy_check_mark:                   |                                            | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             |                                                   |
| `cudatest`   | :heavy_check_mark:                  |                                      |                                            | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             |                                                   |
| `cuda`       | :heavy_check_mark:                  | :heavy_check_mark:                   |                                            | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             |                                                   |
| `cudadev`    | :heavy_check_mark:                  | :heavy_check_mark:                   |                                            | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             |                                                   |
| `cudauvm`    | :heavy_check_mark:                  | :heavy_check_mark:                   |                                            | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             |                                                   |
| `cudacompat` | :heavy_check_mark:                  | :heavy_check_mark:                   |                                            | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             |                                                   |
| `hiptest`    | :heavy_check_mark:                  |                                      |                                            | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             |                                                   |
| `hip`        | :heavy_check_mark:                  | :heavy_check_mark:                   |                                            | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             |                                                   |
| `kokkostest` | :heavy_check_mark:                  |                                      | :heavy_check_mark:                         | :heavy_check_mark:                  |                                                  | :heavy_check_mark:                                             | :heavy_check_mark: (2)                            |
| `kokkos`     | :heavy_check_mark:                  | :heavy_check_mark:                   | :heavy_check_mark:                         |                                     |                                                  |                                                                | :heavy_check_mark: (2)                            |
| `alpakatest` | :heavy_check_mark:                  |                                      |                                            | :heavy_check_mark:                  | :heavy_check_mark:                               |                                                                |                                                   |
| `alpaka`     | :heavy_check_mark:                  |                                      |                                            | :heavy_check_mark:                  | :heavy_check_mark:                               |                                                                |                                                   |
| `sycltest`   | :heavy_check_mark:                  |                                      |                                            |                                     |                                                  |                                                                |                                                   |


1. Boost libraries from the system can also be used, but they need to be version 1.73.0 or newer
2. `kokkos` and `kokkostest` have an optional dependence on hwloc, by default it is not required (see [`kokkos` and `kokkostest`](#kokkos-and-kokkostest) for more details)

The input data set consists of a minimal binary dump of 1000 events of
ttbar+PU events from of
[/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DR-PUAvg50IdealConditions_IdealConditions_102X_upgrade2018_design_v9_ext1-v2/FEVTDEBUGHLT](http://opendata.cern.ch/record/12303)
dataset from the [CMS](https://cms.cern/)
[Open Data](http://opendata.cern.ch/docs/about-cms). The data are
downloaded automatically during the build process.

### Newer GCC versions

RHEL 7.x / CentOS 7.x use GCC 4.8 as their system compiler.
More recent versions can be used from the "Developer Toolset" software collections:
```bash
# list available software collections
$ scl -l
devtoolset-9

# load the GCC 9.x environment
$ source scl_source enable devtoolset-9
$ gcc --version
gcc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

Various versions of GCC are also available from the SFT CVMFS area, for example:
```bash
$ source /cvmfs/sft.cern.ch/lcg/contrib/gcc/8.3.0/x86_64-centos7/setup.sh
$ $ gcc --version
gcc (GCC) 8.3.0
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

RHEL 8.x / CentOS 8.x use GCC 8 as their system compiler.


## Status

| Application  | Description                      | Framework          | Device framework   | Test code          | Raw2Cluster        | RecHit             | Pixel tracking     | Vertex             | Transfers to CPU   | Validation code    | Validated          |
|--------------|----------------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| `fwtest`     | Framework test                   | :heavy_check_mark: |                    | :heavy_check_mark: |                    |                    |                    |                    |                    |                    |                    |
| `serial`     | CPU version (via `cudaCompat`)   | :heavy_check_mark: |                    |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `cudatest`   | CUDA FW test                     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |                    |                    |                    |                    |                    |                    |
| `cuda`       | CUDA version (frozen)            | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `cudadev`    | CUDA version (development)       | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `cudauvm`    | CUDA version with managed memory | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `cudacompat` | `cudaCompat` version             | :heavy_check_mark: | :heavy_check_mark: |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :heavy_check_mark: |
| `hiptest`    | HIP FW test                      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |                    |                    |                    |                    |                    |                    |
| `hip`        | HIP version                      | :heavy_check_mark: | :heavy_check_mark: |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |
| `kokkostest` | Kokkos FW test                   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |                    |                    |                    |                    |                    |                    |
| `kokkos`     | Kokkos version                   | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `alpakatest` | Alpaka FW test                   | :heavy_check_mark: |                    | :white_check_mark: |                    |                    |                    |                    |                    |                    |                    |
| `alpaka`     | Alpaka version                   | :white_check_mark: |                    |                    | :white_check_mark: |                    |                    |                    |                    |                    |                    |
| `sycltest`   | SYCL/oneAPI FW test              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |                    |                    |                    |                    |                    |                    |

The "Device framework" refers to a mechanism similar to [`cms::cuda::Product`](src/cuda/CUDACore/Product.h) and [`cms::cuda::ScopedContext`](src/cuda/CUDACore/ScopedContext.h) to support chains of modules to use the same device and the same work queue.

The column "Validated" means that the program produces the same histograms as the reference `cuda` program within numerical precision (judged "by eye").

## Quick recipe

```bash
# Build application using all available CPUs
$ make -j`nproc` cuda

# For CUDA installations elsewhere than /usr/local/cuda
$ make -j`nproc` cuda CUDA_BASE=/path/to/cuda

# Source environment
$ source env.sh

# Process 1000 events in 1 thread
$ ./cuda

# Command line arguments
$ ./cuda -h
./cuda: [--numberOfThreads NT] [--numberOfStreams NS] [--maxEvents ME] [--data PATH] [--transfer] [--validation] [--empty]

Options
 --numberOfThreads   Number of threads to use (default 1)
 --numberOfStreams   Number of concurrent events (default 0=numberOfThreads)
 --maxEvents         Number of events to process (default -1 for all events in the input file)
 --data              Path to the 'data' directory (default 'data' in the directory of the executable)
 --transfer          Transfer results from GPU to CPU (default is to leave them on GPU)
 --validation        Run (rudimentary) validation at the end (implies --transfer)
 --empty             Ignore all producers (for testing only)
```

### Additional make targets

Note that the contents of `all`, `test`, and all `test_<arch>` targets
are filtered based on the availability of compilers/toolchains. Essentially
* by default programs using only GCC (or "host compiler") are included
* if `CUDA_BASE` directory exists, programs using CUDA are included
* if `SYCL_BASE` directory exists, programs using SYCL are included

| Target                  | Description                                             |
|-------------------------|---------------------------------------------------------|
| `all` (default)         | Build all programs                                      |
| `print_targets`         | Print the programs that would be built with `all`       |
| `test`                  | Run all tests                                           |
| `test_cpu`              | Run tests that use only CPU                             |
| `test_nvidiagpu`        | Run tests that require NVIDIA GPU                       |
| `test_amdgpu`           | Run tests that require AMD GPU                          |
| `test_intelgpu`         | Run tests that require Intel GPU                        |
| `test_auto`             | Run tests that auto-discover the available hardware     |
| `test_<program>`        | Run tests for program `<program>`                       |
| `test_<program>_<arch>` | Run tests for program `<program>` that require `<arch>` |
| `format`                | Format the code with `clang-format`                     |
| `clean`                 | Remove all build artifacts                              |
| `distclean`             | `clean` and remove all externals                        |
| `dataclean`             | Remove downloaded data files                            |
| `external_kokkos_clean` | Remove Kokkos build and installation directory          |

### Test program specific notes (if any)

#### `fwtest`

The printouts can be disabled at compile with with
```
make fwtest ... USER_CXXFLAGS="-DFWTEST_SILENT"
```

#### `serial`

This program is a fork of `cudacompat` by removing all dependencies to
CUDA in order to be a "pure CPU" version. Note that the name refers to
(the absence of) intra-algorithm parallelization and is thus
comparable to the Serial backend of Alpaka or Kokkos. The event-level
parallelism is implemented as in `fwtest`.

#### `cudatest`

The use of caching allocator can be disabled at compile time setting the
`CUDATEST_DISABLE_CACHING_ALLOCATOR` preprocessor symbol:
```
make cudatest ... USER_CXXFLAGS="-DCUDATEST_DISABLE_CACHING_ALLOCATOR"
```

If the caching allocator is disabled and CUDA version is 11.2 or greater is detected,
device allocations and deallocations will use the stream-ordered CUDA functions
`cudaMallocAsync` and `cudaFreeAsync`. Their use can be disabled explicitly at
compile time setting also the `CUDATEST_DISABLE_ASYNC_ALLOCATOR` preprocessor symbol:

```
make cudatest ... USER_CXXFLAGS="-DCUDATEST_DISABLE_CACHING_ALLOCATOR -DCUDATEST_DISABLE_ASYNC_ALLOCATOR"
```

#### `cuda`

This program is frozen to correspond to CMSSW_11_2_0_pre8_Patatrack.

The location of CUDA 11 libraries can be set with `CUDA_BASE` variable.

The use of caching allocator can be disabled at compile time setting the
`CUDA_DISABLE_CACHING_ALLOCATOR` preprocessor symbol:
```
make cuda ... USER_CXXFLAGS="-DCUDA_DISABLE_CACHING_ALLOCATOR"
```

If the caching allocator is disabled and CUDA version is 11.2 or greater is detected,
device allocations and deallocations will use the stream-ordered CUDA functions
`cudaMallocAsync` and `cudaFreeAsync`. Their use can be disabled explicitly at
compile time setting also the `CUDA_DISABLE_ASYNC_ALLOCATOR` preprocessor symbol:

```
make cuda ... USER_CXXFLAGS="-DCUDA_DISABLE_CACHING_ALLOCATOR -DCUDA_DISABLE_ASYNC_ALLOCATOR"
```

#### `cudadev`

This program corresponds to the updated version of the pixel tracking software integrated in
[CMSSW_12_0_0_pre3](https://github.com/cms-sw/cmssw/tree/CMSSW_12_0_0_pre3).

The use of caching allocator can be disabled at compile time setting the
`CUDADEV_DISABLE_CACHING_ALLOCATOR` preprocessor symbol:
```
make cudadev ... USER_CXXFLAGS="-DCUDADEV_DISABLE_CACHING_ALLOCATOR"
```

If the caching allocator is disabled and CUDA version is 11.2 or greater is detected,
device allocations and deallocations will use the stream-ordered CUDA functions
`cudaMallocAsync` and `cudaFreeAsync`. Their use can be disabled explicitly at
compile time setting also the `CUDADEV_DISABLE_ASYNC_ALLOCATOR` preprocessor symbol:

```
make cudadev ... USER_CXXFLAGS="-DCUDADEV_DISABLE_CACHING_ALLOCATOR -DCUDADEV_DISABLE_ASYNC_ALLOCATOR"
```

#### `cudauvm`

The purpose of this program is to test the performance of the CUDA
managed memory. There are various macros that can be used to switch on
and off various behaviors. The default behavior is to use use managed
memory only for those memory blocks that are used for memory
transfers, call `cudaMemPrefetchAsync()`, and
`cudaMemAdvise(cudaMemAdviseSetReadMostly)`. The macros can be set at
compile time along
```
make cudauvm ... USER_CXXFLAGS="-DCUDAUVM_DISABLE_ADVISE"
```

| Macro                                  | Effect                                                |
|----------------------------------------|-------------------------------------------------------|
| `-DCUDAUVM_DISABLE_ADVISE`             | Disable `cudaMemAdvise(cudaMemAdviseSetReadMostly)`   |
| `-DCUDAUVM_DISABLE_PREFETCH`           | Disable `cudaMemPrefetchAsync`                        |
| `-DCUDAUVM_DISABLE_CACHING_ALLOCATOR`  | Disable caching allocator                             |
| `-DCUDAUVM_MANAGED_TEMPORARY`          | Use managed memory also for temporary data structures |
| `-DCUDAUVM_DISABLE_MANAGED_BEAMSPOT`   | Disable managed memory in `BeamSpotToCUDA`            |
| `-DCUDAUVM_DISABLE_MANAGED_CLUSTERING` | Disable managed memory in `SiPixelRawToClusterCUDA`   |
| `-DCUDAUVM_DISABLE_MANAGED_RECHIT`     | Disable managed memory in `SiPixelRecHitCUDA`         |
| `-DCUDAUVM_DISABLE_MANAGED_TRACK `     | Disable managed memory in `CAHitNtupletCUDA`          |
| `-DCUDAUVM_DISABLE_MANAGED_VERTEX `    | Disable managed memory in `PixelVertexProducerCUDA`   |

To use managed memory also for temporary device-only allocations, compile with
```
make cudauvm ... USER_CXXFLAGS="-DCUDAUVM_MANAGED_TEMPORARY"
```

#### `cudacompat`

This program is a fork of `cuda` by extending the use of `cudaCompat` to clustering and RecHits. The aim is to run the same code on CPU. Currently, however, the program requires a GPU because of (still) using pinned host memory in a few places. In the future the program could be extended to provide both CUDA and CPU flavors.

The program contains the changes from following external PRs on top of `cuda`
* [cms-patatrack/cmssw#586](https://github.com/cms-patatrack/cmssw/pull/586)
* [cms-patatrack/cmssw#588](https://github.com/cms-patatrack/cmssw/pull/588)


#### `hip` and `hiptest`

The path to ROCm can be set with `ROCM_BASE` variable.

Note that `hip` does not currently run.


#### `kokkos` and `kokkostest`

```bash
# If nvcc is not in $PATH, create environment file and source it
$ make environment [CUDA_BASE=...]
$ source env.sh

# Actual build command
$ make -j N kokkos [CUDA_BASE=...] [KOKKOS_CUDA_ARCH=...] [...]
$ ./kokkos --cuda

# If changing KOKKOS_HOST_PARALLEL or KOKKOS_DEVICE_PARALLEL, clean up existing build first
$ make clean external_kokkos_clean
$ make kokkos ...
```
* Note that if `CUDA_BASE` needs to be set, it needs to be set for both `make` commands.
* The target CUDA architecture needs to be set explicitly with `KOKKOS_CUDA_ARCH` (see table below)
* The CMake executable can be set with `CMAKE` in case the default one is too old.
* The backends to be used in the Kokkos runtime library build are set with `KOKKOS_HOST_PARALLEL` and `KOKKOS_DEVICE_PARALLEL` (see table below)
   * The Serial backend is always enabled
* When running, the backend(s) need to be set explicitly via command line parameters
   * `--serial` for CPU serial backend
   * `--pthread` for CPU pthread backend
   * `--cuda` for CUDA backend
   * `--hip` for HIP backend
* Use of multiple threads (`--numberOfThreads`) has not been tested and likely does not work correctly. Concurrent events (`--numberOfStreams`) works.
* Support for HIP backend is still work in progress
  * `kokkostest` runs
  * `kokkos` fails at run time inside the "Pixel tracking"
  * Target AMD GPU architecture needs to be set explicitly with `KOKKOS_HIP_ARCH` (see table below)

| Make variable                         | Description                                                                                                                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CMAKE`                               | Path to CMake executable (by default assume `cmake` is found in `$PATH`))                                                                                                   |
| `KOKKOS_HOST_PARALLEL`                | Host-parallel backend (default empty, possible values: empty, `PTHREAD`)                                                                                                    |
| `KOKKOS_DEVICE_PARALLEL`              | Device-parallel backend (default `CUDA`, possible values: empty, `CUDA`, `HIP`)                                                                                             |
| `CUDA_BASE`                           | Path to CUDA installation. Relevant only if `KOKKOS_DEVICE_PARALLEL=CUDA`.                                                                                                  |
| `KOKKOS_CUDA_ARCH`                    | Target CUDA architecture for Kokkos build (default: `70`, possible values: `50`, `70`, `75`; trivial to extend). Relevant only if `KOKKOS_DEVICE_PARALLEL=CUDA`.            |
| `ROCM_BASE`                           | Path to ROCm installation. Relevant only if `KOKKOS_DEVICE_PARALLEL=HIP`.                                                                                                   |
| `KOKKOS_HIP_ARCH`                     | Target AMD GPU architecture for Kokkos build (default: `VEGA900`, possible values: `VEGA900`, `VEGA909`; trivial to extend). Relevant only if `KOKKOS_DEVICE_PARALLEL=HIP`. |
| `KOKKOS_KOKKOS_PTHREAD_DISABLE_HWLOC` | If defined, do not use hwloc. Relevant only if `KOKKOS_HOST_PARALLEL=PTHREAD`.                                                                                              |

| Macro                                  | Effect                                                            |
|----------------------------------------|-------------------------------------------------------------------|
| `-DKOKKOS_SERIALONLY_DISABLE_ATOMICS`  | Disable Kokkos (real) atomics, can be used with Serial-only build |


#### `alpaka`

The `alpaka` code base is loosely based on the `cuda` code base, with some minor changes introduced during the porting.

The use of caching allocator can be disabled at compile time setting the
`ALPAKA_DISABLE_CACHING_ALLOCATOR` preprocessor symbol:
```
make alpaka ... USER_CXXFLAGS="-DALPAKA_DISABLE_CACHING_ALLOCATOR"
```

If the caching allocator is disabled and CUDA version is 11.2 or greater is detected,
device allocations and deallocations will use the stream-ordered CUDA functions
`cudaMallocAsync` and `cudaFreeAsync`. Their use can be disabled explicitly at
compile time setting also the `ALPAKA_DISABLE_ASYNC_ALLOCATOR` preprocessor symbol:

```
make alpaka ... USER_CXXFLAGS="-DALPAKA_DISABLE_CACHING_ALLOCATOR -DALPAKA_DISABLE_ASYNC_ALLOCATOR"
```

## Code structure

The project is split into several programs, one (or more) for each
test case. Each test case has its own directory under [`src`](src)
directory. A test case contains the full application: framework, data
formats, device tooling, plugins for the algorithmic modules ran
by the framework, and the executable.

Each test program is structured as follows within `src/<program name>`
(examples point to [`cuda`](src/cuda)
* [`Makefile`](src/cuda/Makefile) that defines the actual build rules for the program
* [`Makefile.deps`](src/cuda/Makefile.deps) that declares the external dependencies of the program, and the dependencies between shared objects within the program
* [`plugins.txt`](src/cuda/plugins.txt) contains a simple mapping from module names to the plugin shared object names
  - In CMSSW such information is generated automatically by `scram`, in this project the original author was lazy to automate that
* [`bin/`](src/cuda/bin/) directory that contains all the framework code for the executable binary. These files should not need to be modified, except [`main.cc`](src/cuda/bin/main.cc) for changin the set of modules to run, and possibly more command line options
* `plugin-<PluginName>/` directories contain the source code for plugins. The `<PluginName>` part specifies the name of the plugin, and the resulting shared object file is `plugin<PluginName>.so`. Note that no other library or plugin may depend on a plugin (either at link time or even thourgh `#includ`ing a header). The plugins may only be loaded through the names of the modules by the [`PluginManager`](src/cuda/bin/PluginManager.h).
* `<LibraryName>/`: the remaining directories are for libraries. The `<LibraryName>` specifies the name of the library, and the resulting shared object file is `lib<LibraryName>.so`. Other libraries or plugins may depend on a library, in which case the dependence must be declared in [`Makefile.deps`](src/cuda/Makefile.deps).
  * [`CondFormats/`](src/cuda/CondFormats/):
  * [`CUDADataFormats/`](src/cuda/CUDADataFormats/): CUDA-specific data structures that can be passed from one module to another via the `edm::Event`. A given portability technology likely needs its own data format directory, the `CUDADataFormats` can be used as an example.
  * [`CUDACore/`](src/cuda/CUDACore/): Various tools for CUDA. A given portability technology likely needs its own tool directory, the `CUDACore` can be used as an example.
  * [`DataFormats/`](src/cuda/DataFormats/): mainly CPU-side data structures that can be passed from one module to another via the `edm::Event`. Some of these are produced by the [`edm::Source`](src/cuda/bin/Source.h) by reading the binary dumps. These files should not need to be modified. New classes may be added, but they should be independent of the portability technology.
  * [`Framework/`](src/cuda/Framework/): crude approximation of the CMSSW framework. Utilizes TBB tasks to orchestrate the processing of the events by the modules. These files should not need to be modified.
  * [`Geometry/`](src/cuda/Geometry/): geometry information, essentially handful of compile-time constants. May be modified.

For more detailed description of the application structure (mostly plugins) see
[CodeStructure.md](doc/CodeStructure.md)

## Build system

The build system is based on pure GNU Make. There are two levels of
Makefiles. The [top-level Makefile](Makefile) handles the building of
the entire project: it defines general build flags, paths to external
dependencies in the system, recipes to download and build the
externals, and targets for the test programs.

For more information see [BuildSystem.md](doc/BuildSystem.md).

## Contribution guide

Given that the approach of this project is to maintain many programs
in a single branch, in order to keep the commit history readable, each
commit should contain changes only for one test program, and the short
commit message should start with the program name, e.g. `[cuda]`. A
pull request may touch many test programs. General commits (e.g.
top-level Makefile or documentation) can be left without such a prefix.

When starting work for a new portability technology, the first steps
are to figure out the installation of the necessary external software
packages and the build rules (both can be adjusted later). It is
probably best to start by cloning the `fwtest` code for the new
program (e.g. `footest` for a technology `foo`), adjust the test
modules to exercise the API of the technology (see `cudatest` for
examples), and start crafting the tools package (`CUDACore` in
`cuda`). 

Pull requests are expected to build (`make all` succeeds) and pass
tests (`make test`). Programs to have build errors should primarily be
filtered out from `$(TARGETS)`, and failing tests should primarily be
removed from the set of tests run by default. Breakages can, however,
be accepted for short periods of time with a good justification.

The code is formatted with `clang-format` version 10.
