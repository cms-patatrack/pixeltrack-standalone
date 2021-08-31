#ifndef HeterogeneousCore_CUDACore_Context_h
#define HeterogeneousCore_CUDACore_Context_h

#include "CUDACore/Product.h"
#include "Framework/WaitingTaskWithArenaHolder.h"
#include "Framework/Event.h"
#include "Framework/EDGetToken.h"
#include "Framework/EDPutToken.h"
#include "CUDACore/EventCache.h"
#include "CUDACore/SharedEventPtr.h"
#include "CUDACore/SharedStreamPtr.h"

namespace cms::cuda {
  namespace impl {
    // This class is intended to be derived by other Context*, not for general use
    class Context {
    public:
      Context(Context const&) = delete;
      Context& operator=(Context const&) = delete;
      Context(Context&&) = delete;
      Context& operator=(Context&&) = delete;

      int device() const { return currentDevice_; }

      // cudaStream_t is a pointer to a thread-safe object, for which a
      // mutable access is needed even if the Context itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      cudaStream_t stream() {
        if (not isInitialized()) {
          initialize();
        }
        return stream_.get();
      }
      const SharedStreamPtr& streamPtr() {
        if (not isInitialized()) {
          initialize();
        }
        return stream_;
      }

    protected:
      // The constructors set the current device, but the device
      // is not set back to the previous value at the destructor. This
      // should be sufficient (and tiny bit faster) as all CUDA API
      // functions relying on the current device should be called from
      // the scope where this context is. The current device doesn't
      // really matter between modules (or across TBB tasks).
      explicit Context(edm::StreamID streamID);

      explicit Context(int device);

      // meant only for testing
      explicit Context(int device, SharedStreamPtr stream);

      bool isInitialized() const { return bool(stream_); }

      void initialize();
      void initialize(const ProductBase& data);

    private:
      int currentDevice_ = -1;
      SharedStreamPtr stream_;
    };

    class ContextGetterBase : public Context {
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
      ContextGetterBase(Args&&... args) : Context(std::forward<Args>(args)...) {}

    private:
      void synchronizeStreams(int dataDevice, cudaStream_t dataStream, bool available, cudaEvent_t dataEvent);
    };

    class ContextHolderHelper {
    public:
      ContextHolderHelper(edm::WaitingTaskWithArenaHolder waitingTaskHolder, int device)
          : waitingTaskHolder_{std::move(waitingTaskHolder)}, device_{device} {}

      template <typename F>
      void pushNextTask(F&& f);

      void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
        waitingTaskHolder_ = std::move(waitingTaskHolder);
      }

      void enqueueCallback(cudaStream_t stream);

    private:
      edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
      int device_;
    };
  }  // namespace impl

  /**
     * The aim of this class is to do necessary per-event "initialization" in ExternalWork acquire():
     * - setting the current device
     * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
     * - synchronizing between CUDA streams if necessary
     * Users should not, however, construct it explicitly.
     */
  class AcquireContext : public impl::ContextGetterBase {
  public:
    explicit AcquireContext(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder), device()} {}
    ~AcquireContext() = default;

    template <typename F>
    void pushNextTask(F&& f) {
      holderHelper_.pushNextTask(std::forward<F>(f));
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

    // internal API
    void commit();

  private:
    impl::ContextHolderHelper holderHelper_;
  };

  /**
   * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
   * - setting the current device
   * - synchronizing between CUDA streams if necessary
   * Users should not, however, construct it explicitly.
   */
  class ProduceContext : public impl::ContextGetterBase {
  public:
    explicit ProduceContext(edm::StreamID streamID) : ContextGetterBase(streamID) {}

    ~ProduceContext() = default;

    template <typename T>
    std::unique_ptr<Product<T>> wrap(T data) {
      // make_unique doesn't work because of private constructor
      return std::unique_ptr<Product<T>>(new Product<T>(device(), streamPtr(), event_, std::move(data)));
    }

    template <typename T, typename... Args>
    auto emplace(edm::Event& iEvent, edm::EDPutTokenT<T> token, Args&&... args) {
      return iEvent.emplace(token, device(), streamPtr(), event_, std::forward<Args>(args)...);
    }

    // internal API
    void commit();

  private:
    // This construcor is only meant for testing
    explicit ProduceContext(int device, SharedStreamPtr stream, SharedEventPtr event)
        : ContextGetterBase(device, std::move(stream)), event_{std::move(event)} {}

    // create the CUDA Event upfront to catch possible errors from its creation
    SharedEventPtr event_ = getEventCache().get();
  };

  /**
   * The aim of this class is to do necessary per-task "initialization" tasks created in ExternalWork acquire():
   * - setting the current device
   * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
   */
  class TaskContext : public impl::Context {
  public:
    /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
    explicit TaskContext(int device, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : Context(device), holderHelper_{std::move(waitingTaskHolder), device} {}

    ~TaskContext() = default;

    template <typename F>
    void pushNextTask(F&& f) {
      holderHelper_.pushNextTask(std::forward<F>(f));
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

    // Internal API
    void commit();

  private:
    impl::ContextHolderHelper holderHelper_;
  };

  /**
   * The aim of this class is to do necessary per-event "initialization" in analyze()
   * - setting the current device
   * - synchronizing between CUDA streams if necessary
   * and enforce that those get done in a proper way in RAII fashion.
   */
  class AnalyzeContext : public impl::ContextGetterBase {
  public:
    /// Constructor to (possibly) re-use a CUDA stream
    explicit AnalyzeContext(edm::StreamID streamID) : ContextGetterBase(streamID) {}
  };

  namespace impl {
    template <typename F>
    void ContextHolderHelper::pushNextTask(F&& f) {
      replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{edm::make_waiting_task_with_holder(
          tbb::task::allocate_root(),
          std::move(waitingTaskHolder_),
          [device = device_, func = std::forward<F>(f)](edm::WaitingTaskWithArenaHolder h) {
            func(TaskContext{device, std::move(h)});
          })});
    }
  }  // namespace impl

  template <typename F>
  void runAcquire(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder holder, F func) {
    AcquireContext context(streamID, std::move(holder));
    func(context);
    context.commit();
  }

  template <typename F>
  void runProduce(edm::StreamID streamID, F func) {
    ProduceContext context(streamID);
    func(context);
    context.commit();
  }
}  // namespace cms::cuda

#endif
