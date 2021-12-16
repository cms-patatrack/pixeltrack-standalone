#ifndef DataFormats_Common_Product_h
#define DataFormats_Common_Product_h

#include <any>

namespace edm {
  class ProductBase {
  public:
    ProductBase() = default;  // Needed only for ROOT dictionary generation
    ~ProductBase() = default;

    template <typename M>
    M const& metadata() const {
      return std::any_cast<M const&>(metadata_);
    }

  protected:
    template <typename M>
    explicit ProductBase(M&& metadata) : metadata_(std::forward<M>(metadata)) {}

  private:
    // TODO: replace with something lighter
    std::any metadata_;
  };

  template <typename T>
  class Product : public ProductBase {
  public:
    Product() = default;  // Needed only for ROOT dictionary generation

    template <typename M>
    explicit Product(M&& metadata, T&& data) : ProductBase(std::forward<M>(metadata)), data_(std::forward<T>(data)) {}

    template <typename M, typename... Args>
    explicit Product(M&& metadata, Args&&... args) : ProductBase(std::forward<M>(metadata)), data_(std::forward<Args>(args)...) {}

    Product(const Product&) = delete;
    Product& operator=(const Product&) = delete;
    Product(Product&&) = default;
    Product& operator=(Product&&) = default;

    // TODO: would this benefit from access protection?

    // In CUDA version the ScopedContext* are declared as friends, but
    // here Product<T> is largely hidden. It does open a possible
    // loophole for someone to explicitly read the Product<T> from
    // Event, but that should be easy to stop in code review.
    T const& data() const { return data_; }

  private:
    T data_;  //!
  };
}

#endif
