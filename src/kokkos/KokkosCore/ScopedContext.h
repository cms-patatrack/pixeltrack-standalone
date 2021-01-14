#ifndef KokkosCore_ScopedContext_h
#define KokkosCore_ScopedContext_h

#include "KokkosCore/Product.h"
#include "KokkosCore/ContextState.h"
#include "Framework/Event.h"
#include "Framework/EDGetToken.h"
#include "Framework/EDPutToken.h"

namespace cms {
  namespace kokkos {
    namespace impl {
      template <typename ExecSpace>
      class ScopedContextGetterBase {
      public:
        ExecSpace const& execSpace() const { return execSpaceSpecific_->execSpace(); }

        template <typename T>
        const T& get(const Product<T>& data) {
          execSpaceSpecific_->synchronizeWith(data.template execSpaceSpecific<ExecSpace>());
          return data.data_;
        }

        template <typename T>
        const T& get(const edm::Event& iEvent, edm::EDGetTokenT<Product<T>> token) {
          return get(iEvent.get(token));
        }

      protected:
        ScopedContextGetterBase() : execSpaceSpecific_(std::make_unique<impl::ExecSpaceSpecific<ExecSpace>>()) {}

        explicit ScopedContextGetterBase(const ProductBase& data)
            : execSpaceSpecific_(data.execSpaceSpecific<ExecSpace>().cloneShareStream()) {}

        explicit ScopedContextGetterBase(std::unique_ptr<ExecSpaceSpecific<ExecSpace>> specific)
            : execSpaceSpecific_(std::move(specific)) {}

        ExecSpaceSpecific<ExecSpace>& execSpaceSpecific() { return *execSpaceSpecific_; }

        std::unique_ptr<ExecSpaceSpecific<ExecSpace>> releaseExecSpaceSpecific() {
          return std::move(execSpaceSpecific_);
        }

      private:
        std::unique_ptr<ExecSpaceSpecific<ExecSpace>> execSpaceSpecific_;
      };
    }  // namespace impl

    template <typename ExecSpace>
    class ScopedContextAcquire : public impl::ScopedContextGetterBase<ExecSpace> {
    public:
      explicit ScopedContextAcquire(edm::WaitingTaskWithArenaHolder holder) : waitingTaskHolder_(std::move(holder)) {}
      explicit ScopedContextAcquire(edm::WaitingTaskWithArenaHolder holder, ContextState<ExecSpace>& contextState)
          : waitingTaskHolder_(std::move(holder)), contextState_(&contextState) {}

      explicit ScopedContextAcquire(const ProductBase& data, edm::WaitingTaskWithArenaHolder holder)
          : impl::ScopedContextGetterBase<ExecSpace>(data), waitingTaskHolder_(std::move(holder)) {}
      explicit ScopedContextAcquire(const ProductBase& data,
                                    edm::WaitingTaskWithArenaHolder holder,
                                    ContextState<ExecSpace>& contextState)
          : impl::ScopedContextGetterBase<ExecSpace>(data),
            waitingTaskHolder_(std::move(holder)),
            contextState_(&contextState) {}

      ~ScopedContextAcquire() {
        this->execSpaceSpecific().enqueueCallback(std::move(waitingTaskHolder_));
        if (contextState_) {
          contextState_->set(this->releaseExecSpaceSpecific());
        }
      }

    private:
      edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
      ContextState<ExecSpace>* contextState_ = nullptr;
    };

    template <typename ExecSpace>
    class ScopedContextProduce : public impl::ScopedContextGetterBase<ExecSpace> {
    public:
      ScopedContextProduce() = default;
      explicit ScopedContextProduce(const ProductBase& data) : impl::ScopedContextGetterBase<ExecSpace>(data) {}
      explicit ScopedContextProduce(ContextState<ExecSpace>& contextState)
          : impl::ScopedContextGetterBase<ExecSpace>(contextState.release()) {}

      ~ScopedContextProduce() { this->execSpaceSpecific().recordEvent(); }

      template <typename T>
      std::unique_ptr<Product<T>> wrap(T data) {
        // make_unique doesn't work because of private constructor
        return std::unique_ptr<Product<T>>(new Product<T>(this->execSpaceSpecific().cloneShareAll(), std::move(data)));
      }

      template <typename T, typename... Args>
      auto emplace(edm::Event& iEvent, edm::EDPutTokenT<T> token, Args&&... args) {
        return iEvent.emplace(token, this->execSpaceSpecific().cloneShareAll(), std::forward<Args>(args)...);
      }
    };
  }  // namespace kokkos
}  // namespace cms

#endif
