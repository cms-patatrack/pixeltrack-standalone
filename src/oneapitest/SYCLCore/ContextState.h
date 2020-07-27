#ifndef HeterogeneousCore_SYCLCore_ContextState_h
#define HeterogeneousCore_SYCLCore_ContextState_h

#include <CL/sycl.hpp>

#include <memory>

namespace cms {
  namespace sycl {
    /**
     * The purpose of this class is to deliver the device and SYCL stream
     * information from ExternalWork's acquire() to producer() via a
     * member/StreamCache variable.
     */
    class ContextState {
    public:
      ContextState() = default;
      ~ContextState() = default;

      ContextState(const ContextState&) = delete;
      ContextState& operator=(const ContextState&) = delete;
      ContextState(ContextState&&) = delete;
      ContextState& operator=(ContextState&& other) = delete;

    private:
      friend class ScopedContextAcquire;
      friend class ScopedContextProduce;
      friend class ScopedContextTask;

      void set(::sycl::device device, ::sycl::queue stream) {
        throwIfStream();
        stream_ = stream;
        has_stream_ = true;
      }

      ::sycl::device device() const { return stream_.get_device(); }

      ::sycl::queue stream() const {
        throwIfNoStream();
        return stream_;
      }

      ::sycl::queue releaseStream() {
        throwIfNoStream();
        // This function needs to effectively reset stream_ (i.e. stream_
        // must be empty after this function). This behavior ensures that
        // the ::sycl::queue is not hold for inadvertedly long (i.e. to
        // the next event), and is checked at run time.
        has_stream_ = false;
        return std::move(stream_);
      }

      void throwIfStream() const;
      void throwIfNoStream() const;

      ::sycl::queue stream_;
      bool has_stream_;
    };
  }  // namespace sycl
}  // namespace cms

#endif
