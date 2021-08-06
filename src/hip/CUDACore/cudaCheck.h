#ifndef HeterogeneousCore_CUDAUtilities_cudaCheck_h
#define HeterogeneousCore_CUDAUtilities_cudaCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// Boost headers
#define BOOST_STACKTRACE_USE_BACKTRACE
// The definition of BOOST_NOINLINE is __attribute__((__noinline__))
// and the AMD HIP definition of __noinline__ is __attribute__((noinline))
// The preprocessor mixes these to create an erroneous __attribute__ clause.
// Workaround is to define BOOST_NOINLINE in the correct form directly.
#define BOOST_NOINLINE __attribute__((noinline))
#include <boost/stacktrace.hpp>

// CUDA headers
#include <hip/hip_runtime.h>

namespace cms {
  namespace hip {

    [[noreturn]] inline void abortOnCudaError(const char* file,
                                              int line,
                                              const char* cmd,
                                              const char* error,
                                              const char* message,
                                              const char* description = nullptr) {
      std::ostringstream out;
      out << "\n";
      out << file << ", line " << line << ":\n";
      out << "cudaCheck(" << cmd << ");\n";
      out << error << ": " << message << "\n";
      if (description)
        out << description << "\n";

      out << "\nCurrent stack trace:\n";
      out << boost::stacktrace::stacktrace();
      out << "\n";

      throw std::runtime_error(out.str());
    }

    inline bool cudaCheck_(
        const char* file, int line, const char* cmd, hipError_t result, const char* description = nullptr) {
      if (result == hipSuccess)
        return true;

      const char* error = hipGetErrorName(result);
      const char* message = hipGetErrorString(result);
      abortOnCudaError(file, line, cmd, error, message, description);
      return false;
    }

  }  // namespace hip
}  // namespace cms

#define cudaCheck(ARG, ...) (cms::hip::cudaCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // HeterogeneousCore_CUDAUtilities_cudaCheck_h
