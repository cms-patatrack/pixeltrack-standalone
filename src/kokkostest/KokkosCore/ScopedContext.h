#ifndef KokkosCore_ScopedContext_h
#define KokkosCore_ScopedContext_h

#include "KokkosCore/Product.h"
#include "Framework/Event.h"
#include "Framework/EDGetToken.h"
#include "Framework/EDPutToken.h"

namespace cms {
  namespace kokkos {
    template <typename ExecSpace>
    class ScopedContextProduce {
    public:
      ScopedContextProduce() : execSpaceSpecific_(std::make_unique<impl::ExecSpaceSpecific<ExecSpace>>()) {}
      explicit ScopedContextProduce(const ProductBase& data)
          : execSpaceSpecific_(data.execSpaceSpecific<ExecSpace>().cloneShareStream()) {}

      ~ScopedContextProduce() { execSpaceSpecific_->recordEvent(); }

      ExecSpace const& execSpace() const { return execSpaceSpecific_->execSpace(); }

      template <typename T>
      std::unique_ptr<Product<T>> wrap(T data) {
        // make_unique doesn't work because of private constructor
        return std::unique_ptr<Product<T>>(new Product<T>(execSpaceSpecific_->cloneShareAll(), std::move(data)));
      }

      template <typename T, typename... Args>
      auto emplace(edm::Event& iEvent, edm::EDPutTokenT<T> token, Args&&... args) {
        return iEvent.emplace(token, execSpaceSpecific_->cloneShareAll(), std::forward<Args>(args)...);
      }

    private:
      std::unique_ptr<impl::ExecSpaceSpecific<ExecSpace>> execSpaceSpecific_;
    };
  }  // namespace kokkos
}  // namespace cms

#endif
