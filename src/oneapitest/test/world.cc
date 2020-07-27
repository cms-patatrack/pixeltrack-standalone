#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

/*
DPCT1015:72: Output needs adjustment.
*/
void print(::sycl::nd_item<3> item_ct1, ::sycl::stream stream_ct1) {
  stream_ct1 << "GPU thread %d\n"; }

int main() {
  std::cout << "World from" << std::endl;
  dpct::get_default_queue().submit([&](::sycl::handler &cgh) {
    ::sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(::sycl::nd_range<3>(::sycl::range<3>(1, 1, 4), ::sycl::range<3>(1, 1, 4)),
                     [=](::sycl::nd_item<3> item_ct1) { print(item_ct1, stream_ct1); });
  });
  dpct::get_current_device().queues_wait_and_throw();
  return 0;
}
