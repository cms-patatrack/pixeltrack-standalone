#ifndef HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h
#define HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h

#include <cstdint>

namespace cms {
  namespace kokkos {
    class AtomicPairCounter {
    public:
      using c_type = unsigned long long int;

      KOKKOS_INLINE_FUNCTION AtomicPairCounter() {}
      KOKKOS_INLINE_FUNCTION AtomicPairCounter(c_type i) { counter.ac = i; }

      KOKKOS_INLINE_FUNCTION AtomicPairCounter& operator=(c_type i) {
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

      KOKKOS_INLINE_FUNCTION Counters get() const { return counter.counters; }

      KOKKOS_INLINE_FUNCTION void zero() {
        counter.counters.m = 0;
        counter.counters.n = 0;
      }

      // increment n by 1 and m by i.  return previous value
      KOKKOS_INLINE_FUNCTION Counters add(uint32_t i) {
        c_type c = i;
        c += incr;
        Atomic2 ret;
        ret.ac = Kokkos::atomic_fetch_add(&counter.ac, c);
        return ret.counters;
      }

    private:
      Atomic2 counter;
    };
  }  // namespace kokkos
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h
