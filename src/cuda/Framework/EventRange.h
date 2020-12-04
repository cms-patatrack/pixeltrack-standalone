#ifndef EventRange_h
#define EventRange_h

#include <cassert>
#include <sstream>
#include <stdexcept>

#include "Framework/Event.h"

namespace edm {

  class EventRange {
  public:
    EventRange(Event* begin, Event* end) : begin_(begin), end_(end) {
      assert(begin_);
      assert(end_);
      assert(end_ >= begin_);
    }

    Event* begin() { return begin_; }

    Event const* begin() const { return begin_; }

    Event* end() { return end_; }

    Event const* end() const { return end_; }

    size_t size() const { return end_ - begin_; }

    bool empty() const { return end_ == begin_; }

    Event& at(size_t index) {
      if (index >= size()) {
        std::stringstream msg;
        msg << "EventRange::at() range check failed: index " << index << " is outside the range [0.." << size() << ")";
        throw std::out_of_range(msg.str());
      }
      return begin_[index];
    }

    Event const& at(size_t index) const {
      if (index >= size()) {
        std::stringstream msg;
        msg << "EventRange::at() range check failed: index " << index << " is outside the range [0.." << size() << ")";
        throw std::out_of_range(msg.str());
      }
      return begin_[index];
    }

    Event& operator[](size_t index) { return begin_[index]; }

    Event const& operator[](size_t index) const { return begin_[index]; }

  private:
    Event* begin_;
    Event* end_;
  };

  class ConstEventRange {
  public:
    ConstEventRange(Event const* begin, Event const* end) : begin_(begin), end_(end) {
      assert(begin_);
      assert(end_);
      assert(end_ >= begin_);
    }

    ConstEventRange(EventRange range) : begin_(range.begin()), end_(range.end()) {}

    Event const* begin() const { return begin_; }

    Event const* end() const { return end_; }

    size_t size() const { return end_ - begin_; }

    bool empty() const { return end_ == begin_; }

    Event const& at(size_t index) {
      if (index >= size()) {
        std::stringstream msg;
        msg << "ConstEventRange::at() range check failed: index " << index << " is outside the range [0.." << size()
            << ")";
        throw std::out_of_range(msg.str());
      }
      return begin_[index];
    }

    Event const& operator[](size_t index) const { return begin_[index]; }

  private:
    Event const* begin_;
    Event const* end_;
  };

}  // namespace edm

#endif  // EventRange_h
