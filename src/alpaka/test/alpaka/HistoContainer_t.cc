#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <random>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/HistoContainer.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename T>
void go(const DevHost& host, const DevAcc1& device, Queue& queue) {
  std::mt19937 eng;
  std::uniform_int_distribution<T> rgen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  constexpr unsigned int N = 12000;
  auto v_buf = alpaka::mem::buf::alloc<T, Idx>(host, N);
  auto v = alpaka::mem::view::getPtrNative(v_buf);
  auto v_d = alpaka::mem::buf::alloc<T, Idx>(device, N);
  alpaka::mem::view::copy(queue, v_d, v_buf, N);

  constexpr uint32_t nParts = 10;
  constexpr uint32_t partSize = N / nParts;
 
  using Hist = cms::alpakatools::HistoContainer<T, 128, N, 8 * sizeof(T), uint32_t, nParts>;
  std::cout << "HistoContainer " << (int)(offsetof(Hist, off)) << ' ' << Hist::nbins() << ' ' << Hist::totbins() << ' '
            << Hist::capacity() << ' ' << offsetof(Hist, bins) - offsetof(Hist, off) << ' '
            << (std::numeric_limits<T>::max() - std::numeric_limits<T>::min()) / Hist::nbins() << std::endl;

  
  auto offsets_buf = alpaka::mem::buf::alloc<uint32_t, Idx>(host, nParts + 1);
  auto offsets = alpaka::mem::view::getPtrNative(offsets_buf);
  auto off_d = alpaka::mem::buf::alloc<uint32_t, Idx>(device, nParts + 1);

  auto h_buf = alpaka::mem::buf::alloc<Hist, Idx>(host, 1u);
  auto h = alpaka::mem::view::getPtrNative(h_buf);
  auto h_d = alpaka::mem::buf::alloc<Hist, Idx>(device, 1u);
  alpaka::mem::view::set(queue, h_d, 0, Vec1::all(1u)); // TO DO: this was added!!!!

  for (int it = 0; it < 5; ++it) {

    offsets[0] = 0;
    for (uint32_t j = 1; j < nParts + 1; ++j) {
      offsets[j] = offsets[j - 1] + partSize - 3 * j;
      assert(offsets[j] <= N);
    }

    if (it == 1) {  // special cases...
      offsets[0] = 0;
      offsets[1] = 0;
      offsets[2] = 19;
      offsets[3] = 32 + offsets[2];
      offsets[4] = 123 + offsets[3];
      offsets[5] = 256 + offsets[4];
      offsets[6] = 311 + offsets[5];
      offsets[7] = 2111 + offsets[6];
      offsets[8] = 256 * 11 + offsets[7];
      offsets[9] = 44 + offsets[8];
      offsets[10] = 3297 + offsets[9];
    }

    alpaka::mem::view::copy(queue, off_d, offsets_buf, nParts + 1);

    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);

    if (it == 2) {  // big bin
      for (long long j = 1000; j < 2000; j++)
        v[j] = sizeof(T) == 1 ? 22 : 3456;
    }

    alpaka::mem::view::copy(queue, v_d, v_buf, N);

    fillManyFromVector(alpaka::mem::view::getPtrNative(h_d), 
		       nParts, 
		       alpaka::mem::view::getPtrNative(v_d), 
		       alpaka::mem::view::getPtrNative(off_d), 
		       offsets[10], 
		       256, 
		       device, 
		       queue);
    alpaka::mem::view::copy(queue, h_buf, h_d, 1u);
    alpaka::wait::wait(queue);
    assert(0 == h->off[0]);
    assert(offsets[10] == h->size());

    auto verify = [&](uint32_t i, uint32_t k, uint32_t t1, uint32_t t2) {
      assert(t1 < N);
      assert(t2 < N);
      if (T(v[t1] - v[t2]) <= 0)
        std::cout << "for " << i << ':' << v[k] << " failed " << v[t1] << ' ' << v[t2] << std::endl;
    };

    auto incr = [](auto& k) { return k = (k + 1) % Hist::nbins(); };

    // make sure it spans 3 bins...
    auto window = T(1300);

    for (uint32_t j = 0; j < nParts; ++j) {
      auto off = Hist::histOff(j);
      for (uint32_t i = 0; i < Hist::nbins(); ++i) {
        auto ii = i + off;
        if (0 == h->size(ii))
          continue;
        auto k = *h->begin(ii);
        if (j % 2)
          k = *(h->begin(ii) + (h->end(ii) - h->begin(ii)) / 2);
        auto bk = h->bin(v[k]);
        assert(bk == i);
        assert(k < offsets[j + 1]);
        auto kl = h->bin(v[k] - window);
        auto kh = h->bin(v[k] + window);
        assert(kl != i);
        assert(kh != i);
        // std::cout << kl << ' ' << kh << std::endl;

        auto me = v[k];
        auto tot = 0;
        auto nm = 0;
        bool l = true;
        auto khh = kh;
        incr(khh);
        for (auto kk = kl; kk != khh; incr(kk)) {
          if (kk != kl && kk != kh)
            nm += h->size(kk + off);
          for (auto p = h->begin(kk + off); p < h->end(kk + off); ++p) {
            if (std::min(std::abs(T(v[*p] - me)), std::abs(T(me - v[*p]))) > window) {
            } else {
              ++tot;
            }
          }
          if (kk == i) {
            l = false;
            continue;
          }
          if (l)
            for (auto p = h->begin(kk + off); p < h->end(kk + off); ++p)
              verify(i, k, k, (*p));
          else
            for (auto p = h->begin(kk + off); p < h->end(kk + off); ++p)
              verify(i, k, (*p), k);
        }
        if (!(tot >= nm)) {
          std::cout << "too bad " << j << ' ' << i << ' ' << int(me) << '/' << (int)T(me - window) << '/'
                    << (int)T(me + window) << ": " << kl << '/' << kh << ' ' << khh << ' ' << tot << '/' << nm
                    << std::endl;
        }
        if (l)
          std::cout << "what? " << j << ' ' << i << ' ' << int(me) << '/' << (int)T(me - window) << '/'
                    << (int)T(me + window) << ": " << kl << '/' << kh << ' ' << khh << ' ' << tot << '/' << nm
                    << std::endl;
        assert(!l);
      }
    }
  }
}

int main() {
  const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
  const DevAcc1 device(alpaka::pltf::getDevByIdx<PltfAcc1>(0u));
  Queue queue(device);

  go<int16_t>(host, device, queue);
  go<int8_t>(host, device, queue);

  return 0;
}
