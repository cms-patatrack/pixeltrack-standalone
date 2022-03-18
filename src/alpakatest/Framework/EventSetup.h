#ifndef Framework_EventSetup_h
#define Framework_EventSetup_h

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>

#include "Framework/demangle.h"

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
#ifdef __cpp_lib_unordered_map_try_emplace
      auto succeeded =
          typeToProduct_.try_emplace(std::type_index(typeid(T)), std::make_unique<ESWrapper<T>>(std::move(prod)));
#else
      auto succeeded =
          typeToProduct_.emplace(std::type_index(typeid(T)), std::make_unique<ESWrapper<T>>(std::move(prod)));
#endif
      if (not succeeded.second) {
        throw std::runtime_error(std::string("Product of type ") + demangle<T> + " already exists");
      }
    }

    template <typename T>
    T const& get() const {
      const auto found = typeToProduct_.find(std::type_index(typeid(T)));
      if (found == typeToProduct_.end()) {
        throw std::runtime_error(std::string("Product of type ") + demangle<T> + " is not produced");
      }
      return static_cast<ESWrapper<T> const&>(*(found->second)).product();
    }

  private:
    std::unordered_map<std::type_index, std::unique_ptr<ESWrapperBase>> typeToProduct_;
  };
}  // namespace edm

#endif  // Framework_EventSetup_h
