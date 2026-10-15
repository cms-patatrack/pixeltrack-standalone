#ifndef EventSetup_h
#define EventSetup_h

#include <memory>
#include <typeindex>
#include <unordered_map>

#include <iostream>

namespace edm {
  // This is very different from CMSSW, but (hopefully) good-enough
  // for this test
  class ESWrapperBase {
  public:
    virtual ~ESWrapperBase() = default;
  };

  template <typename T>
  class ESWrapper : public ESWrapperBase {
  public:
    explicit ESWrapper(std::unique_ptr<T> obj) : obj_{std::move(obj)} {}

    T const& product() const { return *obj_; }

  private:
    std::unique_ptr<T> obj_;
  };

  class EventSetup {
  public:
    explicit EventSetup() {}

    template <typename T>
    void put(std::unique_ptr<T> prod) {
      auto succeeded =
          typeToProduct_.try_emplace(std::type_index(typeid(T)), std::make_unique<ESWrapper<T>>(std::move(prod)));
      if (not succeeded.second) {
        throw std::runtime_error(std::string("Product of type ") + typeid(T).name() + " already exists");
      }
    }

    template <typename T>
    T const& get() const {
      const auto found = typeToProduct_.find(std::type_index(typeid(T)));
      if (found == typeToProduct_.end()) {
        throw std::runtime_error(std::string("Product of type ") + typeid(T).name() + " is not produced");
      }
      return static_cast<ESWrapper<T> const&>(*(found->second)).product();
    }

  private:
    std::unordered_map<std::type_index, std::unique_ptr<ESWrapperBase>> typeToProduct_;
  };
}  // namespace edm

#endif
