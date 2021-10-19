#ifndef HeterogeneousCore_AlpakaCore_ScopedContext_h
#define HeterogeneousCore_AlpakaCore_ScopedContext_h

#include <optional>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/ContextState.h"
#include "AlpakaCore/EventCache.h"
#include "AlpakaCore/Product.h"
#include "AlpakaCore/SharedEventPtr.h"
#include "AlpakaCore/SharedStreamPtr.h"
#include "Framework/EDGetToken.h"
#include "Framework/EDPutToken.h"
#include "Framework/Event.h"
#include "Framework/WaitingTaskWithArenaHolder.h"
#include "chooseDevice.h"
#include "AlpakaCore/StreamCache.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::cms::alpakatest {
  class TestScopedContext;
}

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  namespace impl {
    // This class is intended to be derived by other ScopedContext*, not for general use
    class ScopedContextBase {
    public:
      int device() const { return currentDevice_; }

      // cudaStream_t is a pointer to a thread-safe object, for which a
      // mutable access is needed even if the ScopedContext itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      ::ALPAKA_ACCELERATOR_NAMESPACE::Queue& stream() const { return *(stream_.get()); }
      const SharedStreamPtr& streamPtr() const { return stream_; }

    protected:
      // The constructors set the current device, but the device
      // is not set back to the previous value at the destructor. This
      // should be sufficient (and tiny bit faster) as all CUDA API
      // functions relying on the current device should be called from
      // the scope where this context is. The current device doesn't
      // really matter between modules (or across TBB tasks).

      template <typename T_Acc>
      ScopedContextBase(T_Acc acc, const ProductBase& data) : currentDevice_(data.device()) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        cudaSetDevice(currentDevice_);
#endif
        if (data.mayReuseStream()) {
          stream_ = data.streamPtr();
        } else {
          stream_ = getStreamCache().get(acc);
        }
      }

      explicit ScopedContextBase(int device, SharedStreamPtr stream)
          : currentDevice_(device), stream_(std::move(stream)) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        cudaSetDevice(currentDevice_);
#endif
      }

      template <typename T_Acc>
      explicit ScopedContextBase(T_Acc acc, edm::StreamID streamID)
          : currentDevice_(::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::chooseDevice(streamID)) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        cudaSetDevice(currentDevice_);
#endif
        stream_ = getStreamCache().get(acc);
      }

    private:
      int currentDevice_;
      SharedStreamPtr stream_;
    };

    class ScopedContextGetterBase : public ScopedContextBase {
    public:
      template <typename T>
      const T& get(const Product<T>& data) {
        synchronizeStreams(data.device(), data.stream(), data.isAvailable(), data.event());
        return data.data_;
      }

      template <typename T>
      const T& get(const edm::Event& iEvent, edm::EDGetTokenT<Product<T>> token) {
        return get(iEvent.get(token));
      }

    protected:
      template <typename... Args>
      ScopedContextGetterBase(Args&&... args) : ScopedContextBase(std::forward<Args>(args)...) {}

      void synchronizeStreams(int dataDevice,
                              ::ALPAKA_ACCELERATOR_NAMESPACE::Queue& dataStream,
                              bool available,
                              alpaka::Event<::ALPAKA_ACCELERATOR_NAMESPACE::Queue> dataEvent);
    };

    class ScopedContextHolderHelper {
    public:
      ScopedContextHolderHelper(edm::WaitingTaskWithArenaHolder waitingTaskHolder)
          : waitingTaskHolder_{std::move(waitingTaskHolder)} {}

      template <typename F>
      void pushNextTask(F&& f, ContextState const* state);

      void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
        waitingTaskHolder_ = std::move(waitingTaskHolder);
      }

      void enqueueCallback(int device, ::ALPAKA_ACCELERATOR_NAMESPACE::Queue& stream);

    private:
      edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
    };
  }  // namespace impl

  /**
     * The aim of this class is to do necessary per-event "initialization" in ExternalWork acquire():
     * - setting the current device
     * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
     * - synchronizing between CUDA streams if necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
  class ScopedContextAcquire : public impl::ScopedContextGetterBase {
  public:
    /// Constructor to create a new CUDA stream (no need for context beyond acquire())
    template <typename T_Acc>
    explicit ScopedContextAcquire(T_Acc acc, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextGetterBase(acc, streamID), holderHelper_{std::move(waitingTaskHolder)} {}

    // /// Constructor to create a new CUDA stream, and the context is needed after acquire()
    template <typename T_Acc>
    explicit ScopedContextAcquire(T_Acc acc,
                                  edm::StreamID streamID,
                                  edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                  ContextState& state)
        : ScopedContextGetterBase(acc, streamID), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

    // /// Constructor to (possibly) re-use a CUDA stream (no need for context beyond acquire())
    template <typename T_Acc>
    explicit ScopedContextAcquire(T_Acc acc, const ProductBase& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextGetterBase(acc, data), holderHelper_{std::move(waitingTaskHolder)} {}

    // /// Constructor to (possibly) re-use a CUDA stream, and the context is needed after acquire()
    template <typename T_Acc>
    explicit ScopedContextAcquire(T_Acc acc,
                                  const ProductBase& data,
                                  edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                  ContextState& state)
        : ScopedContextGetterBase(acc, data), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

    ~ScopedContextAcquire();

    template <typename F>
    void pushNextTask(F&& f) {
      if (contextState_ == nullptr)
        throwNoState();
      holderHelper_.pushNextTask(std::forward<F>(f), contextState_);
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

  private:
    void throwNoState();

    impl::ScopedContextHolderHelper holderHelper_;
    ContextState* contextState_ = nullptr;
  };

  /**
     * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
     * - setting the current device
     * - synchronizing between CUDA streams if necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
  class ScopedContextProduce : public impl::ScopedContextGetterBase {
  public:
    /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
    explicit ScopedContextProduce(ContextState& state)
        : ScopedContextGetterBase(state.device(), state.releaseStreamPtr()) {}

    template <typename T_Acc>
    explicit ScopedContextProduce(T_Acc acc, const ProductBase& data) : ScopedContextGetterBase(acc, data) {}

    template <typename T_Acc>
    explicit ScopedContextProduce(T_Acc acc, edm::StreamID streamID) : ScopedContextGetterBase(acc, streamID) {}

    /// Record the CUDA event, all asynchronous work must have been queued before the destructor
    ~ScopedContextProduce();

    template <typename T_Acc, typename T>
    std::unique_ptr<Product<T>> wrap(T_Acc acc, T data) {
      // make_unique doesn't work because of private constructor
      return std::unique_ptr<Product<T>>(new Product<T>(device(), streamPtr(), getEvent(acc), std::move(data)));
    }

    template <typename T_Acc, typename T, typename... Args>
    auto emplace(T_Acc acc, edm::Event& iEvent, edm::EDPutTokenT<T> token, Args&&... args) {
      // return iEvent.emplace(token, device(), streamPtr(), getEvent(acc), std::forward<Args>(args)...);
      return iEvent.emplace(token, std::forward<Args>(args)...);
      // TODO
    }

  private:
    friend class ::ALPAKA_ACCELERATOR_NAMESPACE::cms::alpakatest::TestScopedContext;

    explicit ScopedContextProduce(int device, SharedStreamPtr stream)
        : ScopedContextGetterBase(device, std::move(stream)) {}

    template <typename T_Acc>
    auto getEvent(T_Acc acc) {
      return getEventCache().get(acc);
    }

    // create the CUDA Event upfront to catch possible errors from its creation
  };

  /**
     * The aim of this class is to do necessary per-task "initialization" tasks created in ExternalWork acquire():
     * - setting the current device
     * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
  class ScopedContextTask : public impl::ScopedContextBase {
  public:
    /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
    explicit ScopedContextTask(ContextState const* state, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextBase(state->device(), state->streamPtr()),  // don't move, state is re-used afterwards
          holderHelper_{std::move(waitingTaskHolder)},
          contextState_{state} {}

    ~ScopedContextTask();

    template <typename F>
    void pushNextTask(F&& f) {
      holderHelper_.pushNextTask(std::forward<F>(f), contextState_);
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

  private:
    impl::ScopedContextHolderHelper holderHelper_;
    ContextState const* contextState_;
  };

  /**
     * The aim of this class is to do necessary per-event "initialization" in analyze()
     * - setting the current device
     * - synchronizing between CUDA streams if necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
  class ScopedContextAnalyze : public impl::ScopedContextGetterBase {
  public:
    /// Constructor to (possibly) re-use a CUDA stream
    template <typename T_Acc>
    explicit ScopedContextAnalyze(T_Acc acc, const ProductBase& data) : ScopedContextGetterBase(acc, data) {}
  };

  namespace impl {
    template <typename F>
    void ScopedContextHolderHelper::pushNextTask(F&& f, ContextState const* state) {
      replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{
          edm::make_waiting_task_with_holder(tbb::task::allocate_root(),
                                             std::move(waitingTaskHolder_),
                                             [state, func = std::forward<F>(f)](edm::WaitingTaskWithArenaHolder h) {
                                               func(ScopedContextTask{state, std::move(h)});
                                             })});
    }
  }  // namespace impl

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaCore_ScopedContext_h
