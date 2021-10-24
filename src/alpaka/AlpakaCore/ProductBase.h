#ifndef AlpakaDataFormats_Common_ProductBase_h
#define AlpakaDataFormats_Common_ProductBase_h

#include <atomic>
#include <memory>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfigAcc.h"

namespace cms::alpakatools {

  namespace impl {
    template <typename TQueue>
    class ScopedContextBase;
  }

  template <typename TQueue>
  class ScopedContextProduce;

}  // namespace cms::alpakatools

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  /**
     * Base class for all instantiations of CUDA<T> to hold the
     * non-T-dependent members.
     */
  class ProductBase {
  public:
    using Queue = ::ALPAKA_ACCELERATOR_NAMESPACE::Queue;
    using Event = alpaka::Event<Queue>;

    ProductBase() = default;  // Needed only for ROOT dictionary generation
    ~ProductBase();

    ProductBase(const ProductBase&) = delete;
    ProductBase& operator=(const ProductBase&) = delete;
    ProductBase(ProductBase&& other)
        : stream_{std::move(other.stream_)},
          event_{std::move(other.event_)},
          mayReuseStream_{other.mayReuseStream_.load()} {}
    ProductBase& operator=(ProductBase&& other) {
      stream_ = std::move(other.stream_);
      event_ = std::move(other.event_);
      mayReuseStream_ = other.mayReuseStream_.load();
      return *this;
    }

    bool isValid() const { return stream_.get() != nullptr; }
    bool isAvailable() const;

    alpaka::Dev<Queue> device() const { return alpaka::getDev(stream()); }

    // cudaStream_t is a pointer to a thread-safe object, for which a
    // mutable access is needed even if the ::cms::alpakatools::ScopedContext itself
    // would be const. Therefore it is ok to return a non-const
    // pointer from a const method here.
    Queue& stream() const { return *(stream_.get()); }

    // cudaEvent_t is a pointer to a thread-safe object, for which a
    // mutable access is needed even if the ::cms::alpakatools::ScopedContext itself
    // would be const. Therefore it is ok to return a non-const
    // pointer from a const method here.
    Event& event() const { return *(event_.get()); }

  protected:
    explicit ProductBase(std::shared_ptr<Queue> stream, std::shared_ptr<Event> event)
        : stream_{std::move(stream)}, event_{std::move(event)} {}

  private:
    friend class cms::alpakatools::impl::ScopedContextBase<Queue>;
    friend class cms::alpakatools::ScopedContextProduce<Queue>;

    // The following function is intended to be used only from ScopedContext
    const std::shared_ptr<Queue>& streamPtr() const { return stream_; }

    bool mayReuseStream() const {
      bool expected = true;
      bool changed = mayReuseStream_.compare_exchange_strong(expected, false);
      // If the current thread is the one flipping the flag, it may
      // reuse the stream.
      return changed;
    }

    // The cudaStream_t is really shared among edm::Event products, so
    // using shared_ptr also here
    std::shared_ptr<Queue> stream_;  //!
    // shared_ptr because of caching in ::cms::alpakatools::EventCache
    std::shared_ptr<Event> event_;  //!

    // This flag tells whether the CUDA stream may be reused by a
    // consumer or not. The goal is to have a "chain" of modules to
    // queue their work to the same stream.
    mutable std::atomic<bool> mayReuseStream_ = true;  //!
  };

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // AlpakaDataFormats_Common_ProductBase_h
