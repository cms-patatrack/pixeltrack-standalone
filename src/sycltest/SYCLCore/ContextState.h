#ifndef HeterogeneousCore_SYCLCore_ContextState_h
#define HeterogeneousCore_SYCLCore_ContextState_h

#include <iostream>
#include <memory>
#include <optional>

#include <sycl/sycl.hpp>

namespace cms {
  namespace sycltools {
    /**
     * The purpose of this class is to deliver the device and queue
     * information from ExternalWork's acquire() to producer() via a
     * member variable.
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

      void set(std::shared_ptr<sycl::queue> stream) {
        throwIfStream();
        stream_ = std::move(stream);
      }

      sycl::device device() const {
        throwIfNoStream();
        return stream_->get_device();
      }

      sycl::queue stream() const {
        throwIfNoStream();
        return *stream_;
      }

      std::shared_ptr<sycl::queue> const& streamPtr() const {
        throwIfNoStream();
        return stream_;
      }

      std::shared_ptr<sycl::queue> releaseStreamPtr() {
        throwIfNoStream();
        // This function needs to effectively reset stream_ (i.e. stream_
        // must be empty after this function). This behavior ensures that
        // the sycl::queue is not hold for inadvertedly long (i.e. to
        // the next event), and is checked at run time.
        return std::move(stream_);
      }

      void throwIfStream() const;
      void throwIfNoStream() const;

      std::shared_ptr<sycl::queue> stream_;
    };
  }  // namespace sycltools
}  // namespace cms

#endif
