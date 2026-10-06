#ifndef Event_h
#define Event_h

#include <memory>
#include <utility>
#include <vector>

#include "Framework/ProductRegistry.h"

// type erasure
namespace edm {
  using StreamID = int;

  class Event;

  class WrapperBase {
  public:
    virtual ~WrapperBase() = default;
  };

  /// Tag selecting Wrapper's parenthesised constructor.
  struct InPlace {};

  template <typename T>
  class Wrapper : public WrapperBase {
  public:
    /// Brace-initialises the payload, which is what an aggregate needs.
    template <typename... Args>
    explicit Wrapper(Args&&... args) : obj_{std::forward<Args>(args)...} {}

    /// Parenthesised construction.  Braces would reach for a container's
    /// initializer_list constructor: Wrapper<std::vector<float>>{n, 0.0f} is a
    /// two-element vector, not n elements, and copying one vector into another
    /// does not compile at all.
    template <typename... Args>
    explicit Wrapper(InPlace, Args&&... args) : obj_(std::forward<Args>(args)...) {}

    T const& product() const { return obj_; }

  private:
    friend class Event;

    /// The deliberately narrow escape hatch behind the zero-copy bindings: a
    /// Python module allocates its output here and then fills it in place,
    /// before any other module can observe it.  Only the contents may be
    /// written -- anything that reallocates invalidates the view handed out.
    ///
    /// Private, with Event the only friend, so that the hatch cannot be opened
    /// from anywhere else: the sole caller is emplaceByIndex, which hands the
    /// reference out before the product is reachable by any other module.
    T& mutableProduct() { return obj_; }

    T obj_;
  };

  class Event {
  public:
    explicit Event(int streamId, int eventId, ProductRegistry const& reg)
        : streamId_(streamId), eventId_(eventId), products_(reg.size()) {}

    // An Event owns its products through unique_ptr, so it was only ever
    // copyable in declaration: the copy constructor existed but would not
    // compile.  Saying so explicitly keeps nanobind from trying to instantiate
    // it when Event is bound, and says what was always true.
    Event(Event const&) = delete;
    Event& operator=(Event const&) = delete;
    Event(Event&&) = default;
    Event& operator=(Event&&) = default;

    StreamID streamID() const { return streamId_; }
    int eventID() const { return eventId_; }

    template <typename T>
    T const& get(EDGetTokenT<T> const& token) const {
      return static_cast<Wrapper<T> const&>(*products_[token.index()]).product();
    }

    template <typename T, typename... Args>
    void emplace(EDPutTokenT<T> const& token, Args&&... args) {
      products_[token.index()] = std::make_unique<Wrapper<T>>(std::forward<Args>(args)...);
    }

    // The index-addressed forms, for the Python bindings, which resolve the
    // type by name at run time and so cannot use the typed tokens.  The index
    // is the one the erased token carries, and the type has already been
    // checked against the product registry when the token was handed out.

    template <typename T>
    T const& getByIndex(unsigned int index) const {
      return static_cast<Wrapper<T> const&>(*products_[index]).product();
    }

    /// Constructs the product in place and returns a mutable reference, so
    /// Python fills the final buffer rather than building something C++ then
    /// has to copy.
    template <typename T, typename... Args>
    T& emplaceByIndex(unsigned int index, Args&&... args) {
      auto wrapper = std::make_unique<Wrapper<T>>(InPlace{}, std::forward<Args>(args)...);
      T& product = wrapper->mutableProduct();
      products_[index] = std::move(wrapper);
      return product;
    }

    bool has(unsigned int index) const { return products_[index] != nullptr; }

  private:
    StreamID streamId_;
    int eventId_;
    std::vector<std::unique_ptr<WrapperBase>> products_;
  };
}  // namespace edm

#endif
