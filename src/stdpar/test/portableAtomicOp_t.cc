#include <algorithm>
#include <ranges>
#include <execution>
#include <memory>
#include <random>
#include <cassert>
#include <iostream>
#include <chrono>

#include "CUDACore/portableAtomicOp.h"

int main() {
  constexpr auto N = 2000000;
  auto result{std::make_unique<int>(std::numeric_limits<int>::max())};
  auto iter{std::views::iota(0, N)};
  auto result_ptr{result.get()};

  std::default_random_engine e1(0);
  std::vector<int> v{};
  v.reserve(N);
  auto data{v.data()};
  std::poisson_distribution poisson_dist(N / 2);
  for (int i = 0; i < 5; ++i) {
    for (const auto i : iter)
      v.push_back(poisson_dist(e1));
    auto min = std::min_element(v.begin(), v.end());

    auto s{std::chrono::steady_clock::now()};
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto iter) {
      cms::cuda::atomicMin(result_ptr, data[iter]);
    });
    auto e{std::chrono::steady_clock::now()};
    std::cout << "parallel exec time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << "ms"
              << std::endl;
    std::cout << "From host: " << *min << ", from GPU: " << *result << std::endl;
    assert(*min == *result);
    *result = std::numeric_limits<int>::max();

    s = std::chrono::steady_clock::now();
    std::for_each(std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto iter) {
      cms::cuda::atomicMin(result_ptr, data[iter]);
    });
    e = std::chrono::steady_clock::now();
    std::cout << "seq exec time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << "ms"
              << std::endl;
    std::cout << "From host: " << *min << ", from std algorithm: " << *result << std::endl;
    assert(*min == *result);
    *result = std::numeric_limits<int>::max();

    v.clear();
  }
}