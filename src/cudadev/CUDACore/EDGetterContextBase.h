#ifndef HeterogeneousCore_CUDACore_EDGetterContextBase_h
#define HeterogeneousCore_CUDACore_EDGetterContextBase_h

#include "CUDACore/FwkContextBase.h"
#include "CUDACore/Product.h"
#include "Framework/EDGetToken.h"

namespace cms::cuda::impl {
  /**
   * This class is a base class for Context classes that should be
   * able to read Event Data products
   */
  class EDGetterContextBase : public FwkContextBase {
  public:
    template <typename T>
    const T& get(const Product<T>& data) {
      if (not isInitialized()) {
        initialize(data);
      }
      synchronizeStreams(data.device(), data.stream(), data.isAvailable(), data.event());
      return data.data_;
    }

    template <typename T>
    const T& get(const edm::Event& iEvent, edm::EDGetTokenT<Product<T>> token) {
      return get(iEvent.get(token));
    }

  protected:
    template <typename... Args>
    EDGetterContextBase(Args&&... args) : FwkContextBase(std::forward<Args>(args)...) {}

  private:
    void synchronizeStreams(int dataDevice, cudaStream_t dataStream, bool available, cudaEvent_t dataEvent);
  };
}

#endif
