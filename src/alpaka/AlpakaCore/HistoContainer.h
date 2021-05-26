#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/AtomicPairCounter.h"
#include "AlpakaCore/alpakastdAlgorithm.h"
#include "AlpakaCore/prefixScan.h"

namespace cms {
  namespace alpakatools {

    struct countFromVector {
      template <typename T_Acc, typename Histo, typename T>
      ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                    Histo *__restrict__ h,
                                    uint32_t nh,
                                    T const *__restrict__ v,
                                    uint32_t const *__restrict__ offsets) const {
        const uint32_t nt = offsets[nh];
        cms::alpakatools::for_each_element_in_grid_strided(acc, nt, [&](uint32_t i) {
          auto off = alpaka_std::upper_bound(offsets, offsets + nh + 1, i);
          assert((*off) > 0);
          int32_t ih = off - offsets - 1;
          assert(ih >= 0);
          assert(ih < int(nh));
          h->count(acc, v[i], ih);
        });
      }
    };

    struct fillFromVector {
      template <typename T_Acc, typename Histo, typename T>
      ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                    Histo *__restrict__ h,
                                    uint32_t nh,
                                    T const *__restrict__ v,
                                    uint32_t const *__restrict__ offsets) const {
        const uint32_t nt = offsets[nh];
        cms::alpakatools::for_each_element_in_grid_strided(acc, nt, [&](uint32_t i) {
          auto off = alpaka_std::upper_bound(offsets, offsets + nh + 1, i);
          assert((*off) > 0);
          int32_t ih = off - offsets - 1;
          assert(ih >= 0);
          assert(ih < int(nh));
          h->fill(acc, v[i], i, ih);
        });
      }
    };

    template <typename Histo>
    ALPAKA_FN_HOST ALPAKA_FN_INLINE __attribute__((always_inline)) void launchZero(
        Histo *__restrict__ h, ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
      uint32_t *poff = (uint32_t *)(char *)(&(h->off));
      auto histoOffView = cms::alpakatools::createDeviceView<typename Histo::Counter>(poff, Histo::totbins());

      alpaka::memset(queue, histoOffView, 0, Histo::totbins());
      alpaka::wait(queue);
    }

    template <typename Histo>
    ALPAKA_FN_HOST ALPAKA_FN_INLINE __attribute__((always_inline)) void launchFinalize(
        Histo *__restrict__ h, ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
      uint32_t *poff = (uint32_t *)(char *)(&(h->off));

      const int num_items = Histo::totbins();

      const unsigned int nthreads = 1024;
      const Vec1 threadsPerBlockOrElementsPerThread(nthreads);
      const unsigned int nblocks = (num_items + nthreads - 1) / nthreads;
      const Vec1 blocksPerGrid(nblocks);

      const WorkDiv1 &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
                          workDiv, multiBlockPrefixScanFirstStep<uint32_t>(), poff, poff, num_items));

      const WorkDiv1 &workDivWith1Block =
          cms::alpakatools::make_workdiv(Vec1::all(1), threadsPerBlockOrElementsPerThread);
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(
              workDivWith1Block, multiBlockPrefixScanSecondStep<uint32_t>(), poff, poff, num_items, nblocks));
    }

    template <typename Histo, typename T>
    ALPAKA_FN_HOST ALPAKA_FN_INLINE __attribute__((always_inline)) void fillManyFromVector(
        Histo *__restrict__ h,
        uint32_t nh,
        T const *__restrict__ v,
        uint32_t const *__restrict__ offsets,
        uint32_t totSize,
        unsigned int nthreads,
        ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
      launchZero(h, queue);

      const unsigned int nblocks = (totSize + nthreads - 1) / nthreads;
      const Vec1 blocksPerGrid(nblocks);
      const Vec1 threadsPerBlockOrElementsPerThread(nthreads);
      const WorkDiv1 &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);

      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDiv, countFromVector(), h, nh, v, offsets));
      launchFinalize(h, queue);

      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<ALPAKA_ACCELERATOR_NAMESPACE::Acc1>(workDiv, fillFromVector(), h, nh, v, offsets));
    }

    struct finalizeBulk {
      template <typename T_Acc, typename Assoc>
      ALPAKA_FN_ACC void operator()(const T_Acc &acc, AtomicPairCounter const *apc, Assoc *__restrict__ assoc) const {
        assoc->bulkFinalizeFill(acc, *apc);
      }
    };

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void forEachInBins(Hist const &hist, V value, int n, Func func) {
      int bs = Hist::bin(value);
      int be = std::min(int(Hist::nbins() - 1), bs + n);
      bs = std::max(0, bs - n);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    // iteratate over bins containing all values in window wmin, wmax
    template <typename Hist, typename V, typename Func>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
      auto bs = Hist::bin(wmin);
      auto be = Hist::bin(wmax);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
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

      static constexpr UT bin(T t) {
        constexpr uint32_t shift = sizeT() - nbits();
        constexpr uint32_t mask = (1 << nbits()) - 1;
        return (t >> shift) & mask;
      }

      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void zero() {
        for (auto &i : off)
          i = 0;
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void add(const T_Acc &acc, CountersOnly const &co) {
        for (uint32_t i = 0; i < totbins(); ++i) {
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, off + i, co.off[i]);
        }
      }

      template <typename T_Acc>
      static ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t atomicIncrement(const T_Acc &acc, Counter &x) {
        return alpaka::atomicOp<alpaka::AtomicAdd>(acc, &x, 1u);
      }

      template <typename T_Acc>
      static ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t atomicDecrement(const T_Acc &acc, Counter &x) {
        return alpaka::atomicOp<alpaka::AtomicSub>(acc, &x, 1u);
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void countDirect(const T_Acc &acc, T b) {
        assert(b < nbins());
        atomicIncrement(acc, off[b]);
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void fillDirect(const T_Acc &acc, T b, index_type j) {
        assert(b < nbins());
        auto w = atomicDecrement(acc, off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t
      bulkFill(const T_Acc &acc, AtomicPairCounter &apc, index_type const *v, uint32_t n) {
        auto c = apc.add(acc, n);
        if (c.m >= nbins())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          bins[c.n + j] = v[j];
        return c.m;
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void bulkFinalize(const T_Acc &acc, AtomicPairCounter const &apc) {
        off[apc.get().m] = apc.get().n;
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void bulkFinalizeFill(const T_Acc &acc, AtomicPairCounter const &apc) {
        auto m = apc.get().m;
        auto n = apc.get().n;

        if (m >= nbins()) {  // overflow!
          off[nbins()] = uint32_t(off[nbins() - 1]);
          return;
        }

        cms::alpakatools::for_each_element_in_grid_strided(acc, totbins(), m, [&](uint32_t i) { off[i] = n; });
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const T_Acc &acc, T t) {
        uint32_t b = bin(t);
        assert(b < nbins());
        atomicIncrement(acc, off[b]);
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const T_Acc &acc, T t, index_type j) {
        uint32_t b = bin(t);
        assert(b < nbins());
        auto w = atomicDecrement(acc, off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const T_Acc &acc, T t, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        atomicIncrement(acc, off[b]);
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const T_Acc &acc, T t, index_type j, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        auto w = atomicDecrement(acc, off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      template <typename T_Acc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void finalize(const T_Acc &acc, Counter *ws = nullptr) {
        assert(off[totbins() - 1] == 0);
        blockPrefixScan(acc, off, totbins(), ws);
        assert(off[totbins() - 1] == off[totbins() - 2]);
      }

      constexpr auto size() const { return uint32_t(off[totbins() - 1]); }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const *begin() const { return bins; }
      constexpr index_type const *end() const { return begin() + size(); }

      constexpr index_type const *begin(uint32_t b) const { return bins + off[b]; }
      constexpr index_type const *end(uint32_t b) const { return bins + off[b + 1]; }

      Counter off[totbins()];
      index_type bins[capacity()];
    };

    template <typename I,        // type stored in the container (usually an index in a vector of the input values)
              uint32_t MAXONES,  // max number of "ones"
              uint32_t MAXMANYS  // max number of "manys"
              >
    using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;

  }  // namespace alpakatools
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
