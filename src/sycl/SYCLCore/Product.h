#ifndef SYCLDataFormats_Common_Product_h
#define SYCLDataFormats_Common_Product_h

#include <memory>

#include "SYCLCore/ProductBase.h"

namespace edm {
  template <typename T>
  class Wrapper;
}

namespace cms {
  namespace sycltools {
    namespace impl {
      class ScopedContextGetterBase;
    }

    /**
     * The purpose of this class is to wrap SYCL data to edm::Event in a
     * way which forces correct use of various utilities.
     *
     * The non-default construction has to be done with cms::sycltools::ScopedContext
     * (in order to properly register the SYCL event).
     *
     * The default constructor is needed only for the ROOT dictionary generation.
     *
     * The SYCL event is in practice needed only for stream-stream
     * synchronization, but someone with long-enough lifetime has to own
     * it. Here is a somewhat natural place. If overhead is too much, we
     * can use them only where synchronization between streams is needed.
     */
    template <typename T>
    class Product : public ProductBase {
    public:
      Product() = default;  // Needed only for ROOT dictionary generation

      Product(const Product&) = delete;
      Product& operator=(const Product&) = delete;
      Product(Product&&) = default;
      Product& operator=(Product&&) = default;

    private:
      friend class impl::ScopedContextGetterBase;
      friend class ScopedContextProduce;
      friend class edm::Wrapper<Product<T>>;

      explicit Product(std::shared_ptr<sycl::queue> stream, sycl::event event, T data)
          : ProductBase(std::move(stream), std::move(event)), data_(std::move(data)) {}

      template <typename... Args>
      explicit Product(std::shared_ptr<sycl::queue> stream, sycl::event event, Args&&... args)
          : ProductBase(std::move(stream), std::move(event)), data_(std::forward<Args>(args)...) {}

      T data_;  //!
    };
  }  // namespace sycltools
}  // namespace cms

#endif
