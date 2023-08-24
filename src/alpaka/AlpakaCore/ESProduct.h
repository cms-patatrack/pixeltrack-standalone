#ifndef HeterogeneousCore_AlpakaCore_ESProduct_h
#define HeterogeneousCore_AlpakaCore_ESProduct_h

#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/config.h"
#include "AlpakaCore/alpaka/devices.h"
#include "AlpakaCore/EventCache.h"

namespace cms::alpakatools {

  template <typename TQueue, typename T>
  class ESProduct {
  public:
    using Queue = TQueue;
    using Event = alpaka::Event<Queue>;
    using Device = alpaka::Dev<Queue>;
    using Platform = alpaka::Platform<Device>;

    ESProduct() : gpuDataPerDevice_(devices<Platform>().size()) {
      for (size_t i = 0; i < gpuDataPerDevice_.size(); ++i) {
        gpuDataPerDevice_[i].m_event =
#if !defined(ALPAKA_ACC_SYCL_ENABLED)
            getEventCache<Event>().get(devices<Platform>()[i]);
#else
            std::make_shared<Event>(devices<Platform>()[i]);
#endif
      }
    }

    ~ESProduct() = default;

    // transferAsync should be a function of (T&, cudaStream_t)
    // which enqueues asynchronous transfers (possibly kernels as well)
    // to the CUDA stream
    template <typename F>
    const T& dataForDeviceAsync(Queue& queue, F transferAsync) const {
      auto device = cms::alpakatools::getDeviceIndex(alpaka::getDev(queue));
      auto& data = gpuDataPerDevice_[device];

      // If GPU data has already been filled, we can return it
      // immediately
      if (not data.m_filled.load()) {
        // It wasn't, so need to fill it
        std::scoped_lock<std::mutex> lk{data.m_mutex};

        if (data.m_filled.load()) {
          // Other thread marked it filled while we were locking the mutex, so we're free to return it
          return *data.m_data;
        }

        if (data.m_fillingStream != nullptr) {
          // Someone else is filling

          // Check first if the recorded event has occurred
          if (alpaka::isComplete(*data.m_event)) {
            // It was, so data is accessible from all CUDA streams on
            // the device. Set the 'filled' for all subsequent calls and
            // return the value
            auto should_be_false = data.m_filled.exchange(true);
            assert(not should_be_false);
            data.m_fillingStream = nullptr;
          } else if (data.m_fillingStream != &queue) {
            // Filling is still going on. For other CUDA stream, add
            // wait on the CUDA stream and return the value. Subsequent
            // work queued on the stream will wait for the event to
            // occur (i.e. transfer to finish).
            alpaka::wait(queue, *data.m_event);
          }
          // else: filling is still going on. But for the same CUDA
          // stream (which would be a bit strange but fine), we can just
          // return as all subsequent work should be enqueued to the
          // same CUDA stream (or stream to be explicitly synchronized
          // by the caller)
        } else {
          // Now we can be sure that the data is not yet on the GPU, and
          // this thread is the first to try that.
          data.m_data = std::move(transferAsync(queue));
          assert(data.m_fillingStream == nullptr);
          data.m_fillingStream = &queue;
          // Record in the cudaStream an event to mark the readiness of the
          // EventSetup data on the GPU, so other streams can check for it
          alpaka::enqueue(queue, *data.m_event);
          // Now the filling has been enqueued to the cudaStream, so we
          // can return the GPU data immediately, since all subsequent
          // work must be either enqueued to the cudaStream, or the cudaStream
          // must be synchronized by the caller
        }
      }
      return *data.m_data;
    }

  private:
    struct Item {
      mutable std::mutex m_mutex;
      mutable std::shared_ptr<Event> m_event;  // guarded by m_mutex
      // non-null if some thread is already filling
      mutable Queue* m_fillingStream = nullptr;    // guarded by m_mutex
      mutable std::atomic<bool> m_filled = false;  // easy check if data has been filled already or not
      mutable std::optional<T> m_data;             // guarded by m_mutex
    };

    std::vector<Item> gpuDataPerDevice_;
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_ESProduct_h
