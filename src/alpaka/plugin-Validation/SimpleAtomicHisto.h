#ifndef SimpleAtomicHisto_h
#define SimpleAtomicHisto_h

#include <atomic>
#include <cassert>
#include <exception>
#include <ostream>
#include <vector>

class SimpleAtomicHisto {
public:
  SimpleAtomicHisto() = default;
  explicit SimpleAtomicHisto(int nbins, float min, float max) : data_(nbins + 2), min_(min), max_(max) {}

  // dirty
  SimpleAtomicHisto(SimpleAtomicHisto&& o) : data_(o.data_.size()), min_(o.min_), max_(o.max_) {}
  SimpleAtomicHisto(SimpleAtomicHisto const& o) : data_(o.data_.size()), min_(o.min_), max_(o.max_) {}

  // thread safe
  void fill(float value) {
    int i;
    if (value < min_) {
      i = 0;
    } else if (value >= max_) {
      i = data_.size() - 1;
    } else {
      i = (value - min_) / (max_ - min_) * (data_.size() - 2);
      // handle rounding near maximum
      if (static_cast<unsigned int>(i) == data_.size() - 2) {
        i = data_.size() - 3;
      }
      if (not(i >= 0 and static_cast<unsigned int>(i) < data_.size() - 2)) {
        throw std::runtime_error("SimpleAtomicHisto::fill(" + std::to_string(value) + "): i " + std::to_string(i) +
                                 " min " + std::to_string(min_) + " max " + std::to_string(max_) + " nbins " +
                                 std::to_string(data_.size() - 2));
      }
      ++i;
    }
    assert(i >= 0 and static_cast<unsigned int>(i) < data_.size());
    data_[i] += 1;
  }

  void dump(std::ostream& os) const {
    os << data_.size() << " " << min_ << " " << max_;
    for (auto const& item : data_) {
      os << " " << item;
    }
  };

private:
  std::vector<std::atomic<int>> data_;
  float min_, max_;
};

inline std::ostream& operator<<(std::ostream& os, SimpleAtomicHisto const& h) {
  h.dump(os);
  return os;
}

#endif
