#ifndef KokkosCore_ScopedContext_h
#define KokkosCore_ScopedContext_h

#include "KokkosCore/Product.h"
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

        ExecSpaceSpecific<ExecSpace>& execSpaceSpecific() { return *execSpaceSpecific_; }

      private:
        std::unique_ptr<ExecSpaceSpecific<ExecSpace>> execSpaceSpecific_;
      };
    }  // namespace impl

    template <typename ExecSpace>
    class ScopedContextAcquire : public impl::ScopedContextGetterBase<ExecSpace> {
    public:
      explicit ScopedContextAcquire(edm::WaitingTaskWithArenaHolder holder) : waitingTaskHolder_(std::move(holder)) {}
      explicit ScopedContextAcquire(const ProductBase& data, edm::WaitingTaskWithArenaHolder holder)
          : impl::ScopedContextGetterBase<ExecSpace>(data), waitingTaskHolder_(std::move(holder)) {}

      ~ScopedContextAcquire() { this->execSpaceSpecific().enqueueCallback(std::move(waitingTaskHolder_)); }

    private:
      edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
    };

    template <typename ExecSpace>
    class ScopedContextProduce : public impl::ScopedContextGetterBase<ExecSpace> {
    public:
      ScopedContextProduce() = default;
      explicit ScopedContextProduce(const ProductBase& data) : impl::ScopedContextGetterBase<ExecSpace>(data) {}

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
