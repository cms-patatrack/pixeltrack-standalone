#ifndef Event_h
#define Event_h

#include <memory>
#include <utility>
#include <vector>

#include "Framework/ProductRegistry.h"

// type erasure
namespace edm {
  class Event;

  class StreamID {
  public:
    ~StreamID() = default;
    StreamID() = delete;
    StreamID(const StreamID&) = default;
    StreamID& operator=(const StreamID&) = default;

    bool operator==(const StreamID& iID) const { return iID.value_ == value_; }

    operator unsigned int() const { return value_; }

    /** \return value ranging from 0 to one less than max number of streams.
     */
    unsigned int value() const { return value_; }

    static StreamID invalidStreamID() { return StreamID(0xFFFFFFFFU); }

  private:
    ///Only a Event is allowed to create one of these
    friend class Event;
    explicit StreamID(unsigned int iValue) : value_(iValue) {}

    unsigned int value_;
  };

  class WrapperBase {
  public:
    virtual ~WrapperBase() = default;
  };

  template <typename T>
  class Wrapper : public WrapperBase {
  public:
    template <typename... Args>
    explicit Wrapper(Args&&... args) : obj_{std::forward<Args>(args)...} {}

    T const& product() const { return obj_; }

  private:
    T obj_;
  };

  class Event {
  public:
    explicit Event(int streamId, int eventId, ProductRegistry const& reg)
        : streamId_(streamId), eventId_(eventId), products_(reg.size()) {}

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
