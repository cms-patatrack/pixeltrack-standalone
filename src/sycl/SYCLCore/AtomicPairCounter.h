#ifndef HeterogeneousCore_SYCLUtilities_interface_AtomicPairCounter_h
#define HeterogeneousCore_SYCLUtilities_interface_AtomicPairCounter_h

#include <cstdint>
#include <SYCLCore/syclAtomic.h>

namespace cms {
  namespace sycltools {

    class AtomicPairCounter {
    public:
      using c_type = unsigned long long int;

      AtomicPairCounter() {}
      AtomicPairCounter(c_type i) { counter.ac = i; }

      AtomicPairCounter& operator=(c_type i) {
        counter.ac = i;
        return *this;
      }

      struct Counters {
        uint32_t n;  // in a "One to Many" association is the number of "One"
        uint32_t m;  // in a "One to Many" association is the total number of associations
      };

      union Atomic2 {
        Counters counters;
        c_type ac;
      };

      static constexpr c_type incr = 1UL << 32;

      Counters get() const { return counter.counters; }

      // increment n by 1 and m by i.  return previous value
      __attribute__((always_inline)) Counters add(uint32_t i) {
        c_type c = i;
        c += incr;
        Atomic2 ret;

        ret.ac = cms::sycltools::atomic_fetch_add<cms::sycltools::AtomicPairCounter::c_type>(&counter.ac, c);

        return ret.counters;
      }

    private:
      Atomic2 counter;
    };

  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLUtilities_interface_AtomicPairCounter_h
