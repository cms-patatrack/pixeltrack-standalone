#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "KokkosCore/AtomicPairCounter.h"
#include "KokkosCore/hintLightWeight.h"
#include "KokkosCore/memoryTraits.h"
#include "KokkosCore/atomic.h"
#include "KokkosCore/kokkos_assert.h"

namespace cms {
  namespace kokkos {
    template <typename T, typename ExecSpace>
    KOKKOS_INLINE_FUNCTION uint32_t upper_bound(const Kokkos::View<uint32_t const*, ExecSpace, Restrict>& offsets,
                                                const uint32_t& upper_index,
                                                const T& value) {
      for (uint32_t j = 0; j < upper_index; ++j) {
        if (offsets(j) > value) {
          return j;
        }
      }
      return 0;
    }

    template <typename Histo, typename ExecSpace, typename T>
    KOKKOS_INLINE_FUNCTION void countFromVector(const Kokkos::View<Histo, ExecSpace, Restrict>& h,
                                                const uint32_t nh,
                                                const Kokkos::View<T const*, ExecSpace, Restrict>& v,
                                                const Kokkos::View<uint32_t const*, ExecSpace, Restrict> offsets,
                                                const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
      uint32_t first = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank();
      uint32_t total_threads = teamMember.league_size() * teamMember.team_size();
      for (uint32_t i = first, nt = offsets(nh); i < nt; i += total_threads) {
        uint32_t index = upper_bound(offsets, nh + 1, i);
        assert(offsets(index) > 0);
        int32_t ih = &offsets(index) - &offsets() - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        h().count(v(i), ih);
      }
    }

    template <typename Histo, typename ExecSpace, typename T>
    KOKKOS_INLINE_FUNCTION void fillFromVector(const Kokkos::View<Histo, ExecSpace, Restrict>& h,
                                               const uint32_t nh,
                                               const Kokkos::View<T const*, ExecSpace, Restrict>& v,
                                               const Kokkos::View<uint32_t const*, ExecSpace, Restrict>& offsets,
                                               const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
      int first = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank();
      int total_threads = teamMember.league_size() * teamMember.team_size();

      for (uint32_t i = first, nt = offsets(nh); i < nt; i += total_threads) {
        uint32_t index = 0;
        for (uint32_t j = 0; j <= nh; ++j) {
          if (offsets(j) > i) {
            index = j;
            break;
          }
        }
        assert(offsets(index) > 0);
        int32_t ih = &offsets(index) - &offsets() - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        h().fill(v(i), i, ih);
      }
    }

    template <typename Histo, typename ExecSpace>
    inline void launchZero(const Kokkos::View<Histo, ExecSpace, Restrict>& h, ExecSpace const& execSpace) {
      Kokkos::parallel_for(
          "launchZero_view",
          hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, Histo::totbins())),
          KOKKOS_LAMBDA(const size_t i) { h().off[i] = 0; });
    }

    template <typename Histo, typename ExecSpace>
    inline void launchZero(Histo* h, ExecSpace const& execSpace) {
      Kokkos::parallel_for(
          "launchZero_pointer",
          hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, Histo::totbins())),
          KOKKOS_LAMBDA(const size_t i) { h->off[i] = 0; });
    }

    template <typename Histo, typename ExecSpace>
    inline void launchFinalize(const Kokkos::View<Histo, ExecSpace, Restrict>& h, ExecSpace const& execSpace) {
      Kokkos::parallel_scan(
          "launchFinalize",
          hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, Histo::totbins())),
          KOKKOS_LAMBDA(const int& i, float& upd, const bool& final) {
            upd += h().off[i];
            if (final)
              h().off[i] = upd;
          });
    }

    template <typename Histo, typename ExecSpace, typename T>
    inline void fillManyFromVector(const Kokkos::View<Histo, ExecSpace, Restrict>& h,
                                   const uint32_t nh,
                                   const Kokkos::View<T const*, ExecSpace, Restrict>& v,
                                   const Kokkos::View<uint32_t const*, ExecSpace, Restrict>& offsets,
                                   const uint32_t totSize,
                                   const int nthreads,
                                   ExecSpace const& execSpace) {
      launchZero(h, execSpace);
      using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
      // TODO: spreadin the total amount of work (totSize) manually to
      // the teams in a way that depends on the number of threads per
      // team feels suboptimal.
      //
      // Kokkos::AUTO() for the number of threads does not really work
      // because the number of blocks depends on the number of threads.
      //
      // Maybe this would really be a case for RangePolicy?
#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
      const auto nblocks = (totSize + ExecSpace::impl_thread_pool_size()) / ExecSpace::impl_thread_pool_size();
      auto tp = hintLightWeight(TeamPolicy(execSpace, nblocks, ExecSpace::impl_thread_pool_size()));
#else
      const auto nblocks = (totSize + nthreads - 1) / nthreads;
      auto tp = hintLightWeight(TeamPolicy(execSpace, nblocks, nthreads));
#endif
      Kokkos::parallel_for(
          "countFromVector", tp, KOKKOS_LAMBDA(typename TeamPolicy::member_type const& teamMember) {
            countFromVector(h, nh, v, offsets, teamMember);
          });
      launchFinalize(h, execSpace);
      Kokkos::parallel_for(
          "fillFromVector", tp, KOKKOS_LAMBDA(typename TeamPolicy::member_type const& teamMember) {
            fillFromVector(h, nh, v, offsets, teamMember);
          });
    }

    template <typename Assoc, typename ExecSpace>
    void finalizeBulk(Kokkos::View<AtomicPairCounter, ExecSpace, Restrict> const& apc,
                      Kokkos::View<Assoc, ExecSpace, Restrict> const& assoc,
                      ExecSpace const& execSpace) {
      Kokkos::parallel_for(
          "finalizeBulk",
          hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, Assoc::totbins())),
          KOKKOS_LAMBDA(const int& i) { assoc().bulkFinalizeFill(apc, i); });
    }

    template <typename Assoc, typename ExecSpace>
    void finalizeBulk(Kokkos::View<AtomicPairCounter, ExecSpace, Restrict> const& apc,
                      Assoc* __restrict__ assoc,
                      ExecSpace const& execSpace) {
      Kokkos::parallel_for(
          "finalizeBulk",
          hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, Assoc::totbins())),
          KOKKOS_LAMBDA(const int& i) { assoc->bulkFinalizeFill(apc, i); });
    }

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
    KOKKOS_FORCEINLINE_FUNCTION void forEachInBins(Hist const* hist, V value, int n, Func func) {
      int bs = Hist::bin(value);
      int be = std::min(int(Hist::nbins() - 1), bs + n);
      bs = std::max(0, bs - n);
      assert(be >= bs);
      for (auto pj = hist->begin(bs); pj < hist->end(be); ++pj) {
        func(*pj);
      }
    }

    // iteratate over bins containing all values in window wmin, wmax
    template <typename Histo, typename V, typename Func, typename ExecSpace>
    KOKKOS_FORCEINLINE_FUNCTION void forEachInWindow(const Kokkos::View<Histo*, ExecSpace, Restrict>& hist,
                                                     V wmin,
                                                     V wmax,
                                                     Func const& func) {
      auto bs = Histo::bin(wmin);
      auto be = Histo::bin(wmax);
      assert(be >= bs);
      for (auto pj = hist(0).begin(bs); pj < hist(0).end(be); ++pj) {
        func(*pj);
      }
    }

    template <typename T,                  // the type of the discretized input values
              uint32_t NBINS,              // number of bins
              uint32_t SIZE,               // max number of element
              uint32_t S = sizeof(T) * 8,  // number of significant bits in T
              typename I = uint32_t,  // type stored in the container (usually an index in a vector of the input values)
              uint32_t NHISTS = 1     // number of histos stored
              >
    class HistoContainer {
    public:
      using Counter = uint32_t;

      using CountersOnly = HistoContainer<T, NBINS, 0, S, I, NHISTS>;

      using index_type = I;
      using UT = typename std::make_unsigned<T>::type;

      static constexpr uint32_t ilog2(uint32_t v) {
        constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
        constexpr uint32_t s[] = {1, 2, 4, 8, 16};

        uint32_t r = 0;  // result of log2(v) will go here
        for (auto i = 4; i >= 0; i--)
          if (v & b[i]) {
            v >>= s[i];
            r |= s[i];
          }
        return r;
      }

      static constexpr uint32_t sizeT() { return S; }
      static constexpr uint32_t nbins() { return NBINS; }
      static constexpr uint32_t nhists() { return NHISTS; }
      static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
      static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }
      static constexpr uint32_t capacity() { return SIZE; }

      static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

      KOKKOS_INLINE_FUNCTION static size_t wsSize() {
#ifdef TODO  //__CUDACC__
        uint32_t* v = nullptr;
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v, v, totbins());
        return temp_storage_bytes;
#else
        return 0;
#endif
      }

      static constexpr UT bin(T t) {
        constexpr uint32_t shift = sizeT() - nbits();
        constexpr uint32_t mask = (1 << nbits()) - 1;
        return (t >> shift) & mask;
      }

      KOKKOS_INLINE_FUNCTION void zero() {
        for (auto& i : off)
          i = 0;
      }

      KOKKOS_FORCEINLINE_FUNCTION void add(const CountersOnly& co) {
        for (uint32_t i = 0; i < totbins(); ++i) {
          cms::kokkos::atomic_add(off + i, co.off[i]);
        }
      }

      static KOKKOS_FORCEINLINE_FUNCTION uint32_t atomicIncrement(Counter& x) {
        return cms::kokkos::atomic_fetch_add(&x, 1U);
      }

      static KOKKOS_FORCEINLINE_FUNCTION uint32_t atomicDecrement(Counter& x) {
        return cms::kokkos::atomic_fetch_sub(&x, 1U);
      }

      KOKKOS_FORCEINLINE_FUNCTION void countDirect(T b) {
        assert(b < nbins());
        atomicIncrement(off[b]);
      }

      KOKKOS_FORCEINLINE_FUNCTION void fillDirect(T b, index_type j) {
        assert(b < nbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      template <typename ExecSpace>
      KOKKOS_FORCEINLINE_FUNCTION int32_t bulkFill(Kokkos::View<AtomicPairCounter, ExecSpace, Restrict> apc,
                                                   index_type const* v,
                                                   uint32_t n) {
        auto c = apc().add(n);
        if (c.m >= nbins())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          bins[c.n + j] = v[j];
        return c.m;
      }

      KOKKOS_INLINE_FUNCTION int32_t bulkFill(AtomicPairCounter& apc, index_type const* v, uint32_t n) {
        auto c = apc.add(n);
        if (c.m >= nbins())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          bins[c.n + j] = v[j];
        return c.m;
      }

      template <typename ExecSpace>
      KOKKOS_FORCEINLINE_FUNCTION void bulkFinalize(Kokkos::View<AtomicPairCounter, ExecSpace, Restrict> const& apc) {
        off[apc().get().m] = apc().get().n;
      }

      template <typename ExecSpace>
      KOKKOS_FORCEINLINE_FUNCTION void bulkFinalizeFill(Kokkos::View<AtomicPairCounter, ExecSpace, Restrict> const& apc,
                                                        const int threadId) {
        auto m = apc().get().m;
        auto n = apc().get().n;
        if (m >= nbins()) {  // overflow!
          off[nbins()] = uint32_t(off[nbins() - 1]);
          return;
        }
        auto i = m + threadId;
        if (i < totbins()) {
          off[i] = n;
        }
      }

      KOKKOS_FORCEINLINE_FUNCTION void count(T t) {
        uint32_t b = bin(t);
        assert(b < nbins());
        atomicIncrement(off[b]);
      }

      KOKKOS_FORCEINLINE_FUNCTION void fill(T t, index_type j) {
        uint32_t b = bin(t);
        assert(b < nbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      KOKKOS_FORCEINLINE_FUNCTION void count(T t, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        atomicIncrement(off[b]);
      }

      KOKKOS_FORCEINLINE_FUNCTION void fill(T t, index_type j, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

#pragma hd_warning_disable
      // The team size must be power-of-two
      template <typename TeamMemberType>
      KOKKOS_FORCEINLINE_FUNCTION void finalize(const TeamMemberType& teamMember) {
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(teamMember, totbins()),
                              [=](int i, uint32_t& update, const bool final) {
                                update += this->off[i];
                                if (final)
                                  this->off[i] = update;
                              });
      }

      constexpr auto size() const { return uint32_t(off[totbins() - 1]); }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const* begin() const { return bins; }
      constexpr index_type const* end() const { return begin() + size(); }

      constexpr index_type const* begin(uint32_t b) const { return bins + off[b]; }
      constexpr index_type const* end(uint32_t b) const { return bins + off[b + 1]; }

      Counter off[totbins()];
      index_type bins[capacity()];
    };

    template <typename I,        // type stored in the container (usually an index in a vector of the input values)
              uint32_t MAXONES,  // max number of "ones"
              uint32_t MAXMANYS  // max number of "manys"
              >
    using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;
  }  // namespace kokkos
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
