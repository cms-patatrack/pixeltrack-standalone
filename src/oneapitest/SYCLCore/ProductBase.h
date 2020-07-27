#ifndef SYCLDataFormats_Common_ProductBase_h
#define SYCLDataFormats_Common_ProductBase_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <atomic>
#include <memory>

namespace cms {
  namespace sycl {
    namespace impl {
      class ScopedContextBase;
    }

    /**
     * Base class for all instantiations of SYCL<T> to hold the
     * non-T-dependent members.
     */
    class ProductBase {
    public:
      ProductBase() = default;  // Needed only for ROOT dictionary generation
      ~ProductBase();

      ProductBase(const ProductBase&) = delete;
      ProductBase& operator=(const ProductBase&) = delete;
      ProductBase(ProductBase&& other)
          : stream_{std::move(other.stream_)},
            event_{std::move(other.event_)},
            mayReuseStream_{other.mayReuseStream_.load()},
            device_{other.device_} {}
      ProductBase& operator=(ProductBase&& other) {
        stream_ = std::move(other.stream_);
        event_ = std::move(other.event_);
        mayReuseStream_ = other.mayReuseStream_.load();
        device_ = other.device_;
        return *this;
      }

      bool isValid() const { return true; }
      bool isAvailable() const;

      ::sycl::device device() const { return *device_; }

      // ::sycl::queue is (hopefully) a thread-safe object, for which a
      // mutable access is needed even if the cms::sycl::ScopedContext itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      ::sycl::queue stream() const { return *stream_; }

      // ::sycl::event is (hopefully) a thread-safe object, for which a
      // mutable access is needed even if the cms::sycl::ScopedContext itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      ::sycl::event event() const { return *event_; }

    protected:
      explicit ProductBase(::sycl::device device, ::sycl::queue stream, ::sycl::event event)
          : stream_{std::move(stream)}, event_{std::move(event)}, device_{device} {}

    private:
      friend class impl::ScopedContextBase;
      friend class ScopedContextProduce;

      // The following function is intended to be used only from ScopedContext
      const ::sycl::queue& streamPtr() const { return *stream_; }

      bool mayReuseStream() const {
        bool expected = true;
        bool changed = mayReuseStream_.compare_exchange_strong(expected, false);
        // If the current thread is the one flipping the flag, it may
        // reuse the stream.
        return changed;
      }

      std::optional<::sycl::queue> stream_;  //!
      std::optional<::sycl::event> event_;  //!

      // This flag tells whether the SYCL stream may be reused by a
      // consumer or not. The goal is to have a "chain" of modules to
      // queue their work to the same stream.
      mutable std::atomic<bool> mayReuseStream_ = true;  //!

      // The SYCL device associated with this product
      std::optional<::sycl::device> device_;  //!
    };
  }  // namespace sycl
}  // namespace cms

#endif
