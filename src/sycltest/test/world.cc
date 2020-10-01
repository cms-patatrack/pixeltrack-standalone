#include <iostream>
#include <CL/sycl.hpp>

void sycl_exception_handler(cl::sycl::exception_list exceptions) {
  std::ostringstream msg;
  msg << "Caught asynchronous SYCL exception:";
  for (auto const &exc_ptr : exceptions) {
    try {
      std::rethrow_exception(exc_ptr);
    } catch (cl::sycl::exception const &e) {
      msg << '\n' << e.what();
    }
    throw std::runtime_error(msg.str());
  }
}

void print(sycl::nd_item<1> item, sycl::stream out) {
  out << "SYCL device thread " << item.get_local_id(0) << sycl::endl;
}

int main() {
  std::cout << "World from" << std::endl;

  sycl::queue queue{sycl::default_selector(), sycl_exception_handler, sycl::property::queue::in_order()};
  queue.submit([&](sycl::handler &cgh) {
    sycl::stream out(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(4), sycl::range<1>(4)),
                     [=](sycl::nd_item<1> item) {
                       print(item, out);
                     });
  });
  queue.wait_and_throw();
  return 0;
}
