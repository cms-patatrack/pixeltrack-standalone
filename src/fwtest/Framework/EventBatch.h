#ifndef EventBatch_h
#define EventBatch_h

#include <vector>

#include "Framework/Event.h"
#include "Framework/EventRange.h"

namespace edm {

  class EventBatch {
  public:
    Event& emplace(int streamId, int eventId, ProductRegistry const& reg) {
      events_.emplace_back(streamId, eventId, reg);
      return events_.back();
    }

    void reserve(size_t capacity) { events_.reserve(capacity); }

    void clear() { events_.clear(); }

    size_t size() const { return events_.size(); }

    bool empty() const { return events_.empty(); }

    EventRange range() { return EventRange(&events_.front(), &events_.back() + 1); }

    ConstEventRange range() const { return ConstEventRange(&events_.front(), &events_.back() + 1); }

  private:
    std::vector<edm::Event> events_;
  };

}  // namespace edm

#endif  // EventBatch_h
