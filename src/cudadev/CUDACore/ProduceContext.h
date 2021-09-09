#ifndef HeterogeneousCore_CUDACore_ProduceContext_h
#define HeterogeneousCore_CUDACore_ProduceContext_h

#include "CUDACore/EDGetterContextBase.h"
#include "CUDACore/EventCache.h"
#include "Framework/EDPutToken.h"

namespace cms::cuda {
  /**
   * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
   * - setting the current device
   * - synchronizing between CUDA streams if necessary
   * Users should not, however, construct it explicitly.
   */
  class ProduceContext : public impl::EDGetterContextBase {
  public:
    explicit ProduceContext(edm::StreamID streamID) : EDGetterContextBase(streamID) {}

    ~ProduceContext() = default;

    template <typename T>
    std::unique_ptr<Product<T>> wrap(T data) {
      // make_unique doesn't work because of private constructor
      return std::unique_ptr<Product<T>>(new Product<T>(device(), streamSharingHelper(), event_, std::move(data)));
    }

    template <typename T, typename... Args>
    auto emplace(edm::Event& iEvent, edm::EDPutTokenT<T> token, Args&&... args) {
      return iEvent.emplace(token, device(), streamSharingHelper(), event_, std::forward<Args>(args)...);
    }

    // internal API
    void commit();

  private:
    // This construcor is only meant for testing
    explicit ProduceContext(int device, SharedStreamPtr stream, SharedEventPtr event)
        : EDGetterContextBase(device, std::move(stream)), event_{std::move(event)} {}

    // create the CUDA Event upfront to catch possible errors from its creation
    SharedEventPtr event_ = getEventCache().get();
  };

  template <typename F>
  void runProduce(edm::StreamID streamID, F&& func) {
    ProduceContext context(streamID);
    func(context);
    context.commit();
  }
}

#endif
