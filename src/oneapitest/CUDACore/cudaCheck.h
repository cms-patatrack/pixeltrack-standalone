#ifndef HeterogeneousCore_CUDAUtilities_cudaCheck_h
#define HeterogeneousCore_CUDAUtilities_cudaCheck_h

// C++ standard headers
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers

namespace cms {
  namespace cuda {

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
      throw std::runtime_error(out.str());
    }

    inline bool cudaCheck_(
        const char* file, int line, const char* cmd, int result, const char* description = nullptr) try {
      if (result == 0)
        return true;

      /*
      DPCT1009:2: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
      */
      const char* error = "cudaGetErrorName not supported" /*cudaGetErrorName(result)*/;
      /*
      DPCT1009:3: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
      */
      const char* message = "cudaGetErrorString not supported" /*cudaGetErrorString(result)*/;
      abortOnCudaError(file, line, cmd, error, message, description);
      return false;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

  }  // namespace cuda
}  // namespace cms

#define cudaCheck(ARG, ...) (cms::cuda::cudaCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // HeterogeneousCore_CUDAUtilities_cudaCheck_h
