#ifndef HeterogeneousCore_CUDAUtilities_requireDevices_h
#define HeterogeneousCore_CUDAUtilities_requireDevices_h

/**
 * These functions are meant to be called only from unit tests.
 */
namespace cms {
  namespace hiptest {
    /// In presence of CUDA devices, return true; otherwise print message and return false
    bool testDevices();

    /// Print message and exit if there are no CUDA devices
    void requireDevices();
  }  // namespace hiptest
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_requireDevices_h
