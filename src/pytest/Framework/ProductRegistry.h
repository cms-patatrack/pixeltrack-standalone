#ifndef ProductRegistry_h
#define ProductRegistry_h

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>

#include "Framework/EDGetToken.h"
#include "Framework/EDPutToken.h"
#include "Framework/PayloadTypes.h"

namespace edm {
  // A product is identified by its C++ type *and* the label of the module that
  // produces it, so one job may hold several products of the same type -- two
  // implementations of one algorithm, say, to be compared against each other.
  // fwtest keys on the type alone and rejects the second producer of a type.
  //
  // The label is the module's configuration label, not its class name: the same
  // plugin may be instantiated several times under different labels.
  class ProductRegistry {
  public:
    constexpr static int kSourceIndex = 0;
    static char const* kSourceLabel() { return "source"; }

    ProductRegistry() = default;

    // public interface

    /// Declare a product of type T, published under the label of the module
    /// currently being constructed.
    template <typename T>
    EDPutTokenT<T> produces() {
      const Key key{std::type_index(typeid(T)), currentLabel_};
      const unsigned int ind = productToIndex_.size();
      auto succeeded = productToIndex_.try_emplace(key, currentModuleIndex_, ind);
      if (not succeeded.second) {
        throw std::runtime_error(std::string("Product of type ") + typeid(T).name() + " with label '" +
                                 currentLabel_ + "' already exists");
      }
      return EDPutTokenT<T>{ind};
    }

    /// Consume the product of type T published by the module labelled `label`.
    /// An empty label means "the only product of that type", so a module reads
    /// its input label from its configuration and passes it on unconditionally:
    /// the label has to be given only in a job that has more than one candidate.
    template <typename T>
    EDGetTokenT<T> consumes(std::string const& label) {
      if (label.empty()) {
        return consumes<T>();
      }
      const Key key{std::type_index(typeid(T)), label};
      const auto found = productToIndex_.find(key);
      if (found == productToIndex_.end()) {
        throw std::runtime_error(std::string("Product of type ") + typeid(T).name() + " with label '" + label +
                                 "' is not produced by the source or any preceding module");
      }
      consumedModules_.insert(found->second.moduleIndex());
      return EDGetTokenT<T>{found->second.productIndex()};
    }

    /// Consume the product of type T when exactly one module produces that
    /// type.  Ambiguity is an error rather than a silent choice: a module that
    /// wants one of several same-typed products has to name which.
    template <typename T>
    EDGetTokenT<T> consumes() {
      return EDGetTokenT<T>{uniqueProductIndex(std::type_index(typeid(T)), typeid(T).name())};
    }

    // The type-erased forms, for the Python layer.  The registry only ever
    // stores a std::type_index, so these need no template: the payload name is
    // resolved to that index through the EDM_PAYLOAD_TYPES table.  They hand
    // back the same EDPutToken / EDGetToken a C++ module would end up with,
    // carrying the type name so the generated dispatch can find the bindings.
    EDPutToken produces(std::string const& typeName) {
      const Key key{payloadTypeIndex(typeName), currentLabel_};
      const unsigned int ind = productToIndex_.size();
      auto succeeded = productToIndex_.try_emplace(key, currentModuleIndex_, ind);
      if (not succeeded.second) {
        throw std::runtime_error("Product of type " + typeName + " with label '" + currentLabel_ +
                                 "' already exists");
      }
      return EDPutToken{ind, typeName};
    }

    EDGetToken consumes(std::string const& typeName, std::string const& label) {
      const std::type_index type = payloadTypeIndex(typeName);
      if (label.empty()) {
        return EDGetToken{uniqueProductIndex(type, typeName), typeName};
      }
      const Key key{type, label};
      const auto found = productToIndex_.find(key);
      if (found == productToIndex_.end()) {
        throw std::runtime_error("Product of type " + typeName + " with label '" + label +
                                 "' is not produced by the source or any preceding module");
      }
      consumedModules_.insert(found->second.moduleIndex());
      return EDGetToken{found->second.productIndex(), typeName};
    }

    auto size() const { return productToIndex_.size(); }

    // internal interface
    void beginModuleConstruction(int i, std::string label) {
      currentModuleIndex_ = i;
      currentLabel_ = std::move(label);
      consumedModules_.clear();
    }

    std::set<unsigned> const& consumedModules() { return consumedModules_; }

  private:
    /// The product of that type, when exactly one module produces it.  Records
    /// the dependency, as the labelled lookups do.
    unsigned int uniqueProductIndex(std::type_index type, std::string const& typeName) {
      Indices const* match = nullptr;
      unsigned int count = 0;
      for (auto const& entry : productToIndex_) {
        if (entry.first.type == type) {
          ++count;
          match = &entry.second;
        }
      }
      if (count == 0) {
        throw std::runtime_error("Product of type " + typeName + " is not produced");
      }
      if (count > 1) {
        throw std::runtime_error("Product of type " + typeName + " is produced by " + std::to_string(count) +
                                 " modules; name one with consumes<T>(label)");
      }
      consumedModules_.insert(match->moduleIndex());
      return match->productIndex();
    }

    struct Key {
      std::type_index type;
      std::string label;

      bool operator==(Key const& other) const = default;
    };

    struct KeyHash {
      std::size_t operator()(Key const& key) const {
        const std::size_t h1 = std::hash<std::type_index>{}(key.type);
        const std::size_t h2 = std::hash<std::string>{}(key.label);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
      }
    };

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
    std::string currentLabel_ = kSourceLabel();
    std::set<unsigned int> consumedModules_;

    std::unordered_map<Key, Indices, KeyHash> productToIndex_;
  };
}  // namespace edm

#endif
