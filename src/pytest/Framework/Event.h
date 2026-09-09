#ifndef Event_h
#define Event_h

#include <memory>
#include <utility>
#include <vector>

#include "Framework/ProductRegistry.h"

// type erasure
namespace edm {
  using StreamID = int;

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
    T obj_;
  };

  class Event {
  public:
    explicit Event(int streamId, int eventId, ProductRegistry const& reg)
        : streamId_(streamId), eventId_(eventId), products_(reg.size()) {}

    // An Event owns its products through unique_ptr, so it was only ever
    // copyable in declaration: the copy constructor existed but would not
    // compile.  Saying so explicitly says what was always true.
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

  private:
    StreamID streamId_;
    int eventId_;
    std::vector<std::unique_ptr<WrapperBase>> products_;
  };
}  // namespace edm

#endif
