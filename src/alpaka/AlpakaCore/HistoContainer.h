#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>


#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/AtomicPairCounter.h"
#include "AlpakaCore/alpakastdAlgorithm.h"
#include "AlpakaCore/prefixScan.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

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
	const uint32_t gridDimension(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u]);
	const auto& [firstElementIdx, endElementIdx] = cms::alpakatools::element_global_index_range(acc, Vec1::all(nt));
	uint32_t endElementIdxStrided = endElementIdx[0u];
	for (uint32_t threadIndexStrided = firstElementIdx[0u]; threadIndexStrided < nt; threadIndexStrided += gridDimension) {
	  for (uint32_t i = threadIndexStrided; i < endElementIdxStrided; ++i) {
	    auto off = alpaka_std::upper_bound(offsets, offsets + nh + 1, i);
	    assert((*off) > 0);
	    int32_t ih = off - offsets - 1;
	    assert(ih >= 0);
	    assert(ih < int(nh));
	    h->count(acc, v[i], ih);
	  }
	  endElementIdxStrided += gridDimension;
	}
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
	const uint32_t gridDimension(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u]);
	const auto &[firstElementIdx, endElementIdx] = cms::alpakatools::element_global_index_range(acc, Vec1::all(nt));

	uint32_t endElementIdxStrided = endElementIdx[0u];
	for (uint32_t threadIdxStrided = firstElementIdx[0u]; threadIdxStrided < nt; threadIdxStrided += gridDimension) {
	  for (uint32_t i = threadIdxStrided; i < endElementIdxStrided; ++i) {
	    auto off = alpaka_std::upper_bound(offsets, offsets + nh + 1, i);
	    assert((*off) > 0);
	    int32_t ih = off - offsets - 1;
	    assert(ih >= 0);
	    assert(ih < int(nh));
	    h->fill(acc, v[i], i, ih);
	  }
	  endElementIdxStrided += gridDimension;
	}
      }
    };

    struct launchZero {
      template <typename T_Acc, typename Histo>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void operator()(const T_Acc &acc,
										    Histo *__restrict__ h) const {
	//uint32_t *poff = (uint32_t *)((char *)(h) + offsetof(Histo, off));
	//int32_t size = offsetof(Histo, bins) - offsetof(Histo, off);
	//assert(size >= int(sizeof(uint32_t) * Histo::totbins()));

	// TO DO: USE A WORKDIV??????????????
	for (uint32_t i = 0; i < Histo::totbins(); ++i) {
	  h->off[i] = 0;
	}
      }
    };

    /*
    struct multiBlockPrefixScanFirstStepHisto {
      template <typename T_Acc, typename T>
	ALPAKA_FN_ACC void operator()(const T_Acc& acc, Histo *__restrict__ h, T* psum_d, int32_t size) const {
	multiBlockPrefixScanFirstStepHisto<uint32_t>(
	  h->sum, // TO DO: GetPointerNative??
	  h->sum, // TO DO: ppws??
	  psum_d,
	  size));
	  };*/


  template <typename Histo>
      ALPAKA_FN_HOST ALPAKA_FN_INLINE  __attribute__((always_inline)) void launchFinalize(Histo *__restrict__ h,
											  const DevAcc1& device,
											  Queue& queue) {

    uint32_t *poff = (uint32_t *)((char *)(h) + offsetof(Histo, off));
    // NB: Why are we not interested in poff on device memory (cuda version as well, different from test). ??
      
      //int32_t *ppsws = (int32_t *)((char *)(h) + offsetof(Histo, psws)); // now unused???
      // ppsws ?????????????????????????????????????????????????????????????????????????????????


      
    const int num_items = Histo::totbins();

    auto psum_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, Vec1::all(num_items));
    uint32_t* psum_d = alpaka::mem::view::getPtrNative(psum_dBuf);

    const unsigned int nthreads = 1024;
    const unsigned int nblocks = (num_items + nthreads - 1) / nthreads;
    const Vec1 &blocksPerGrid(Vec1::all(nblocks));  
    const Vec1 &threadsPerBlockOrElementsPerThread(Vec1::all(nthreads));

    const WorkDiv1 &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<Acc1>(workDiv,
								  multiBlockPrefixScanFirstStep<uint32_t>(),
								  poff, // TO DO: GetPointerNative??
								  poff, // TO DO: ppws??
								  psum_d,
								  num_items));

    const WorkDiv1 &workDivWith1Block = cms::alpakatools::make_workdiv(Vec1::all(1), threadsPerBlockOrElementsPerThread);
    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<Acc1>(workDivWith1Block,
								  multiBlockPrefixScanSecondStep<uint32_t>(),
								  poff,
								  poff,
								  psum_d,
								  num_items,
								  nblocks));
    }

    template <typename Histo, typename T>
      ALPAKA_FN_HOST ALPAKA_FN_INLINE  __attribute__((always_inline)) void fillManyFromVector(Histo *__restrict__ h,
												  uint32_t nh,
												  T const *__restrict__ v,
												  uint32_t const *__restrict__ offsets,
												  uint32_t totSize,
												  unsigned int nthreads,
												  const DevAcc1& device,
											      Queue& queue) {
      std::cout << "Start within fillManyFromVector" << std::endl;
      alpaka::queue::enqueue(queue,
			     alpaka::kernel::createTaskKernel<Acc1>(WorkDiv1{Vec1::all(1u), Vec1::all(1u), Vec1::all(1u)},
								    launchZero(),
								    h));

      unsigned int nblocks = (totSize + nthreads - 1) / nthreads;
      const Vec1 &blocksPerGrid(Vec1::all(nblocks));  
      const Vec1 &threadsPerBlockOrElementsPerThread(Vec1::all(nthreads));
      const WorkDiv1 &workDiv = cms::alpakatools::make_workdiv(blocksPerGrid, threadsPerBlockOrElementsPerThread);
      alpaka::queue::enqueue(queue,
			     alpaka::kernel::createTaskKernel<Acc1>(workDiv,
								    countFromVector(),
								    h, nh, v, offsets));
     


      launchFinalize(h, device, queue);
      alpaka::queue::enqueue(queue,
			     alpaka::kernel::createTaskKernel<Acc1>(workDiv,
								    fillFromVector(),
								    h, nh, v, offsets));
    }

    struct finalizeBulk {
      template <typename T_Acc, typename Assoc>
	ALPAKA_FN_ACC void operator()(const T_Acc &acc, AtomicPairCounter const *apc, Assoc *__restrict__ assoc) const {
	assoc->bulkFinalizeFill(acc, *apc);
      }
    };

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
      ALPAKA_FN_HOST ALPAKA_FN_INLINE void forEachInBins(Hist const &hist, V value, int n, Func func) {
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
    ALPAKA_FN_HOST ALPAKA_FN_INLINE void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
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
      ALPAKA_FN_HOST_ACC HistoContainer() {}; // TO DO: not neeeded??????????
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

    ALPAKA_FN_HOST ALPAKA_FN_INLINE void zero() {
      for (auto &i : off)
	i = 0;
    }

    /*
    ALPAKA_FN_HOST ALPAKA_FN_INLINE void add(CountersOnly const &co) {
      for (uint32_t i = 0; i < totbins(); ++i) {
	auto &a = (std::atomic<Counter> &)(off[i]);
	a += co.off[i];
      }
      }*/

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void add(const T_Acc& acc, CountersOnly const &co) {
      for (uint32_t i = 0; i < totbins(); ++i) {
	alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, off + i, co.off[i]);
      }
    }

    template <typename T_Acc>
    static ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t atomicIncrement(const T_Acc& acc, Counter &x) {
      return alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &x, 1u);
    }

    template <typename T_Acc>
    static ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t atomicDecrement(const T_Acc& acc, Counter &x) {
      return alpaka::atomic::atomicOp<alpaka::atomic::op::Sub>(acc, &x, 1u);
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void countDirect(const T_Acc& acc, T b) {
      assert(b < nbins());
      atomicIncrement(acc, off[b]);
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fillDirect(const T_Acc& acc, T b, index_type j) {
      assert(b < nbins());
      auto w = atomicDecrement(acc, off[b]);
      assert(w > 0);
      bins[w - 1] = j;
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t bulkFill(const T_Acc& acc, AtomicPairCounter &apc, index_type const *v, uint32_t n) {
      auto c = apc.add(acc, n);
      if (c.m >= nbins())
	return -int32_t(c.m);
      off[c.m] = c.n;
      for (uint32_t j = 0; j < n; ++j)
	bins[c.n + j] = v[j];
      return c.m;
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void bulkFinalize(const T_Acc& acc, AtomicPairCounter const &apc) {
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

      const uint32_t gridDimension(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u]);
      const auto &[firstElementIdx, endElementIdx] = cms::alpakatools::element_global_index_range(acc, Vec1::all(totbins()));

      uint32_t endElementIdxStrided = m + endElementIdx[0u];
      for (uint32_t threadIdxStrided = m + firstElementIdx[0u]; threadIdxStrided < totbins(); threadIdxStrided += gridDimension) {
	for (uint32_t i = threadIdxStrided; i < endElementIdxStrided; ++i) {
	  off[i] = n;
	}
	endElementIdxStrided += gridDimension;
      }
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const T_Acc& acc, T t) {
      uint32_t b = bin(t);
      assert(b < nbins());
      atomicIncrement(acc, off[b]);
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const T_Acc& acc, T t, index_type j) {
      uint32_t b = bin(t);
      assert(b < nbins());
      auto w = atomicDecrement(acc, off[b]);
      assert(w > 0);
      bins[w - 1] = j;
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const T_Acc& acc, T t, uint32_t nh) {
      uint32_t b = bin(t);
      assert(b < nbins());
      b += histOff(nh);
      assert(b < totbins());
      atomicIncrement(acc, off[b]);
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const T_Acc& acc, T t, index_type j, uint32_t nh) {
      uint32_t b = bin(t);
      assert(b < nbins());
      b += histOff(nh);
      assert(b < totbins());
      auto w = atomicDecrement(acc, off[b]);
      assert(w > 0);
      bins[w - 1] = j;
    }

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void finalize(const T_Acc& acc, Counter *ws = nullptr) {
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
      int32_t psws;  // prefix-scan working space
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
