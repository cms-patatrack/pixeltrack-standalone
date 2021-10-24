#ifndef HeterogeneousCore_AlpakaCore_ContextState_h
#define HeterogeneousCore_AlpakaCore_ContextState_h

#include <memory>
#include <optional>

#include "AlpakaCore/alpakaConfig.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  /**
     * The purpose of this class is to deliver the device and CUDA stream
     * information from ExternalWork's acquire() to producer() via a
     * member/StreamCache variable.
     */
  class ContextState {
  public:
    using Device = ::ALPAKA_ACCELERATOR_NAMESPACE::Device;
    using Queue = ::ALPAKA_ACCELERATOR_NAMESPACE::Queue;

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

    void set(Device device, std::shared_ptr<Queue> stream) {
      throwIfStream();
      device_ = std::move(device);
      stream_ = std::move(stream);
    }

    Device const& device() const { return *device_; }

    const std::shared_ptr<Queue>& streamPtr() const {
      throwIfNoStream();
      return stream_;
    }

    std::shared_ptr<Queue> releaseStreamPtr() {
      throwIfNoStream();
      // This function needs to effectively reset stream_ (i.e. stream_
      // must be empty after this function). This behavior ensures that
      // the std::shared_ptr<Queue> is not hold for inadvertedly long (i.e. to
      // the next event), and is checked at run time.
      return std::move(stream_);
    }

    void throwIfStream() const;
    void throwIfNoStream() const;

    std::shared_ptr<Queue> stream_;
    // Work around no default constructor for Device
    // On the other hand, we don't strictly need the Device here...
    std::optional<Device> device_;
  };

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaCore_ContextState_h
