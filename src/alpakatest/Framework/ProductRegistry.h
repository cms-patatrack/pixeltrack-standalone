#ifndef Framework_ProductRegistry_h
#define Framework_ProductRegistry_h

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>

#include "Framework/EDGetToken.h"
#include "Framework/EDPutToken.h"
#include "Framework/demangle.h"

namespace edm {
  class ProductRegistry {
  public:
    constexpr static int kSourceIndex = 0;

    ProductRegistry() = default;

    // public interface
    template <typename T>
    EDPutTokenT<T> produces() {
      const std::type_index ti{typeid(T)};
      const unsigned int ind = typeToIndex_.size();
#ifdef __cpp_lib_unordered_map_try_emplace
      auto succeeded = typeToIndex_.try_emplace(ti, currentModuleIndex_, ind);
#else
      auto succeeded = typeToIndex_.emplace(ti, Indices{currentModuleIndex_, ind});
#endif
      if (not succeeded.second) {
        throw std::runtime_error(std::string("Product of type ") + demangle<T> + " already exists");
      }
      return EDPutTokenT<T>{ind};
    }

    template <typename T>
    EDGetTokenT<T> consumes() {
      const auto found = typeToIndex_.find(std::type_index(typeid(T)));
      if (found == typeToIndex_.end()) {
        throw std::runtime_error(std::string("Product of type ") + demangle<T> + " is not produced");
      }
      consumedModules_.insert(found->second.moduleIndex());
      return EDGetTokenT<T>{found->second.productIndex()};
    }

    auto size() const { return typeToIndex_.size(); }

    // internal interface
    void beginModuleConstruction(int i) {
      currentModuleIndex_ = i;
      consumedModules_.clear();
    }

    std::set<unsigned> const& consumedModules() { return consumedModules_; }

  private:
    class Indices {
    public:
      explicit Indices(unsigned int mi, unsigned int pi) : moduleIndex_(mi), productIndex_(pi) {}

      unsigned int moduleIndex() const { return moduleIndex_; }
      unsigned int productIndex() const { return productIndex_; }

    private:
      unsigned int moduleIndex_;  // index of producing module
      unsigned int productIndex_;
    };

    unsigned int currentModuleIndex_ = kSourceIndex;
    std::set<unsigned int> consumedModules_;

    std::unordered_map<std::type_index, Indices> typeToIndex_;
  };
}  // namespace edm

#endif  // Framework_ProductRegistry_h
