#ifndef AlpakaCore_alpakaFwd_h
#define AlpapaCore_alpakaFwd_h

/**
 * This file forward declares specific types defined in Alpaka
 * (depending on the backend-enabling macros) so that these types
 * would be available throughout CMSSW without a direct dependence on
 * Alpaka in order to avoid the constraints that would impose
 * (primarily the device compiler)
 *
 * This is a little bit brittle, but let's see how it goes.
 */
namespace alpaka {
  // Devices
  class DevCpu;
  class DevUniformCudaHipRt;
  using DevCudaRt = DevUniformCudaHipRt;

  // Queues
  template <typename TDev> class QueueGenericThreadsBlocking;
  using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;
  template<typename TDev> class QueueGenericThreadsNonBlocking;
  using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;
  class QueueUniformCudaHipRtNonBlocking;
  using QueueCudaRtNonBlocking = QueueUniformCudaHipRtNonBlocking;
}

#endif
