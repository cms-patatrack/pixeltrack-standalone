#ifndef HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h
#define HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h

#include <cstdint>

namespace cms {
  namespace alpakatools {

    class AtomicPairCounter {
    public:
      using c_type = unsigned long long int;

      ALPAKA_FN_HOST_ACC AtomicPairCounter() {}
      ALPAKA_FN_HOST_ACC AtomicPairCounter(c_type i) { counter.ac = i; }

      ALPAKA_FN_HOST_ACC AtomicPairCounter& operator=(c_type i) {
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

      ALPAKA_FN_HOST_ACC Counters get() const { return counter.counters; }

      // increment n by 1 and m by i.  return previous value
      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE Counters add(const T_Acc& acc, uint32_t i) {
        c_type c = i;
        c += incr;

        Atomic2 ret;
        ret.ac = alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &counter.ac, c);
        return ret.counters;
      }

    private:
      Atomic2 counter;
    };

  }  // namespace alpakatools
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h
