#ifndef CUDADataFormats_Common_ProductBase_h
#define CUDADataFormats_Common_ProductBase_h

#include <atomic>
#include <memory>

#include "CUDACore/SharedStreamPtr.h"
#include "CUDACore/SharedEventPtr.h"

namespace cms {
  namespace cuda {
    namespace impl {
      class FwkContextBase;

      /**
       * The CUDA stream is shared between all the Event products of
       * the EDProducer. If the stream gets re-used, only one consumer
       * of all the products should be allowed to use the stream. An
       * objects of this class is shared between such Event products
       * and takes care of letting only those consumers get the stream.
       */
      class StreamSharingHelper {
      public:
        explicit StreamSharingHelper(SharedStreamPtr stream) : stream_(std::move(stream)) {}
        StreamSharingHelper(const StreamSharingHelper&) = delete;
        StreamSharingHelper& operator=(const StreamSharingHelper&) = delete;
        StreamSharingHelper(StreamSharingHelper&&) = delete;
        StreamSharingHelper& operator=(StreamSharingHelper) = delete;

        const SharedStreamPtr& streamPtr() const { return stream_; }

        bool mayReuseStream() const {
          bool expected = true;
          bool changed = mayReuseStream_.compare_exchange_strong(expected, false);
          // If the current thread is the one flipping the flag, it may
          // reuse the stream.
          return changed;
        }

      private:
        SharedStreamPtr stream_;

        // This flag tells whether the CUDA stream may be reused by a
        // consumer or not. The goal is to have a "chain" of modules to
        // queue their work to the same stream.
        mutable std::atomic<bool> mayReuseStream_ = true;
      };
    }  // namespace impl

    /**
     * Base class for all instantiations of CUDA<T> to hold the
     * non-T-dependent members.
     */
    class ProductBase {
    public:
      ProductBase() = default;  // Needed only for ROOT dictionary generation
      ~ProductBase();

      ProductBase(const ProductBase&) = delete;
      ProductBase& operator=(const ProductBase&) = delete;
      ProductBase(ProductBase&& other) = default;
      ProductBase& operator=(ProductBase&& other) = default;

      bool isAvailable() const;

      int device() const { return device_; }

      // cudaStream_t is a pointer to a thread-safe object, for which a
      // mutable access is needed even if the ProductBase itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      cudaStream_t stream() const { return stream_->streamPtr().get(); }

      // cudaEvent_t is a pointer to a thread-safe object, for which a
      // mutable access is needed even if the ProductBase itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      cudaEvent_t event() const { return event_.get(); }

    protected:
      explicit ProductBase(int device, std::shared_ptr<impl::StreamSharingHelper> stream, SharedEventPtr event)
          : stream_{std::move(stream)}, event_{std::move(event)}, device_{device} {}

    private:
      friend class impl::FwkContextBase;
      friend class ProduceContext;

      // The following function is intended to be used only from Context
      const SharedStreamPtr& streamPtr() const { return stream_->streamPtr(); }

      bool mayReuseStream() const { return stream_->mayReuseStream(); }

      // Helper shared between all cms::cuda::Product<T> event
      // products of an EDProducer
      std::shared_ptr<impl::StreamSharingHelper> stream_;  //!

      // shared_ptr because of caching in cms::cuda::EventCache
      SharedEventPtr event_;  //!

      // The CUDA device associated with this product
      int device_ = -1;  //!
    };
  }  // namespace cuda
}  // namespace cms

#endif
