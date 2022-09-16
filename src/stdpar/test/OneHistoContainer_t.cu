#include <algorithm>
#include <ranges>
#include <execution>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include "CUDACore/HistoContainer.h"

using namespace cms::cuda;

template <typename T, int NBINS, int S, int DELTA>
void mykernel(T const* __restrict__ v, uint32_t N) {
  assert(v);
  assert(N == 12000);

  printf("start kernel for %d data\n", N);

  using Hist = HistoContainer<T, NBINS, 12000, S, uint16_t>;
  std::unique_ptr<Hist> hist_ptr{std::make_unique<Hist>()};
  Hist *hist = hist_ptr.get();
  std::fill(std::execution::par, hist->off, hist->off + Hist::totbins(), 0);

  auto iter_n{std::views::iota(0U, N)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter_n), std::ranges::cend(iter_n), [=](const auto j){
    hist->count(v[j]);
  });

  assert(0 == hist->size());

  hist->finalize();

  assert(N == hist->size());

  auto iter_nbins{std::views::iota(0U, Hist::nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter_nbins), std::ranges::cend(iter_nbins), [=](const auto j){
    assert(hist->off[j] <= hist->off[j + 1]);
  });

  std::for_each(std::execution::par, std::ranges::cbegin(iter_n), std::ranges::cend(iter_n), [=](const auto j){
    hist->fill(v[j], j);
  });

  assert(0 == hist->off[0]);
  assert(N == hist->size());

  auto iter_hsize{std::views::iota(0U, hist->size() - 1)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter_hsize), std::ranges::cend(iter_hsize), [=](const auto j){
    auto p = hist->begin() + j;
    assert((*p) < N);
    auto k1 = Hist::bin(v[*p]);
    auto k2 = Hist::bin(v[*(p + 1)]);
    assert(k2 >= k1);
  });

  std::for_each(std::execution::par, std::ranges::cbegin(iter_hsize), std::ranges::cend(iter_hsize), [=](const auto i){
    auto p = hist->begin() + i;
    auto j = *p;
    auto b0 = Hist::bin(v[j]);
    int tot = 0;
    auto ftest = [&](int k) {
      assert(k >= 0 && k < N);
      ++tot;
    };
    forEachInWindow(*hist, v[j], v[j], ftest);
    int rtot = hist->size(b0);
    assert(tot == rtot);
    tot = 0;
    auto vm = int(v[j]) - DELTA;
    auto vp = int(v[j]) + DELTA;
    constexpr int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
    vm = std::max(vm, 0);
    vm = std::min(vm, vmax);
    vp = std::min(vp, vmax);
    vp = std::max(vp, 0);
    assert(vp >= vm);
    forEachInWindow(*hist, vm, vp, ftest);
    int bp = Hist::bin(vp);
    int bm = Hist::bin(vm);
    rtot = hist->end(bp) - hist->begin(bm);
    assert(tot == rtot);
  });
}

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go() {
  std::mt19937 eng;

  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::uniform_int_distribution<T> rgen(rmin, rmax);

  constexpr int N = 12000;

  auto v = std::make_unique<T[]>(N);
  assert(v.get());

  using Hist = HistoContainer<T, NBINS, N, S>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::capacity() << ' '
            << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;

    assert(v.get());
    mykernel<T, NBINS, S, DELTA>(v.get(), N);
  }
}

int main() {
  go<int16_t>();
  go<uint8_t, 128, 8, 4>();
  go<uint16_t, 313 / 2, 9, 4>();

  return 0;
}
