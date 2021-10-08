#ifndef HeterogeneousCore_CUDAUtilities_ESProductNew_h
#define HeterogeneousCore_CUDAUtilities_ESProductNew_h

#include "CUDACore/EventCache.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"

#include "CUDACore/eventWorkHasCompleted.h"

#include <any>
#include <mutex>
#include <optional>

namespace cms::cudaNew {
  template <typename T>
  class ESProduct {
  public:
    ESProduct() : gpuDataPerDevice_(cms::cuda::deviceCount()) {}

    std::size_t size() const { return gpuDataPerDevice_.size(); }

    template <typename U>
    void emplace(std::size_t i, U&& data, cudaStream_t stream) {
      auto event = cms::cuda::getEventCache().get();
      cudaCheck(cudaEventRecord(event.get(), stream));
      gpuDataPerDevice_[i].data_.emplace(std::forward<U>(data));
      gpuDataPerDevice_[i].event_ = std::move(event);
    }

    template <typename U>
    void setHostData(U&& data) {
      // TODO: wrapping to shared_ptr to have a copyable type for std::any...
      hostData_ = std::make_shared<U>(std::forward<U>(data));
      hostDataAlive_ = true;
    }

    T const& get(int device, cudaStream_t stream) const {
      auto const& item = gpuDataPerDevice_[device];
      // if the production has completed, we can just return the data
      if (item.complete_) {
        return *item.data_;
      }

      {
        std::scoped_lock<std::mutex> lk{item.mutex_};
        // if some other thread beat us, we can just return the data
        if (item.complete_) {
          return *item.data_;
        }
        // if asynchronous work is still incomplete, insert wait on the stream and return
        if (not cms::cuda::eventWorkHasCompleted(item.event_.get())) {
          cudaCheck(cudaStreamWaitEvent(stream, item.event_.get(), 0), "Failed to make a stream to wait for an event");
          return *item.data_;
        }
        // work was complete, can release the event
        item.complete_ = true;
        item.event_.reset();
      }

      // check if all devices are complete, and if yes, we can release the host data
      if (std::all_of(
              gpuDataPerDevice_.begin(), gpuDataPerDevice_.end(), [](Item const& e) { return e.complete_.load(); })) {
        bool wasAlive = hostDataAlive_.exchange(false);
        if (wasAlive) {
          hostData_.reset();
        }
      }

      return *item.data_;
    }

  private:
    struct Item {
      mutable std::mutex mutex_;
      mutable cms::cuda::SharedEventPtr event_;  // guarded by mutex_
      mutable std::atomic<bool> complete_;
      std::optional<T> data_;  // optional to avoid requiring default constructor in T
    };

    std::vector<Item> gpuDataPerDevice_;
    mutable std::atomic<bool> hostDataAlive_ = false;
    mutable std::any
        hostData_;  // to be kept alive until all asynchronous activity has finished, guarded by AND of complete_
  };
}  // namespace cms::cudaNew

#endif
