#ifndef KokkosCore_Product_h
#define KokkosCore_Product_h

#include "KokkosCore/ProductBase.h"

namespace edm {
  template <typename T>
  class Wrapper;
}

namespace cms {
  namespace kokkos {
    namespace impl {
      template <typename ExecSpace>
      class ScopedContextGetterBase;
    }
    template <typename ExecSpace>
    class ScopedContextProduce;

    template <typename T>
    class Product : public ProductBase {
    public:
      Product() = default;

      Product(const Product&) = delete;
      Product& operator=(const Product&) = delete;
      Product(Product&&) = default;
      Product& operator=(Product&&) = default;

    private:
      template <typename ExecSpace>
      friend class impl::ScopedContextGetterBase;
      template <typename ExecSpace>
      friend class ScopedContextProduce;
      friend class edm::Wrapper<Product<T>>;

      explicit Product(std::unique_ptr<impl::ExecSpaceSpecificBase> spaceSpecific, T data)
          : ProductBase(std::move(spaceSpecific)), data_(std::move(data)) {}

      template <typename... Args>
      explicit Product(std::unique_ptr<impl::ExecSpaceSpecificBase> spaceSpecific, Args&&... args)
          : ProductBase(std::move(spaceSpecific)), data_(std::forward<Args>(args)...) {}

      T data_;
    };
  }  // namespace kokkos
}  // namespace cms

#endif
