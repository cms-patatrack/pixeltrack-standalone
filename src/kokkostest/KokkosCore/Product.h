#ifndef KokkosCore_Product_h
#define KokkosCore_Product_h

#include "KokkosCore/ProductBase.h"

namespace cms {
  namespace kokkos {
    template <typename T>
    class Product : public ProductBase {
    public:
      Product() = default;
      Product(std::unique_ptr<impl::ExecSpaceSpecificBase> spaceSpecific, T data)
          : ProductBase(std::move(spaceSpecific)) {}

      Product(const Product&) = delete;
      Product& operator=(const Product&) = delete;
      Product(Product&&) = default;
      Product& operator=(Product&&) = default;
    };
  }  // namespace kokkos
}  // namespace cms

#endif
