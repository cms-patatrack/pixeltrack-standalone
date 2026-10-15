#include <iostream>
#include <sstream>

#include "plugin-Validation/SimpleAtomicHisto.h"

int main() {
  SimpleAtomicHisto h(10, 0, 1);
  std::stringstream ss;

  ss << h;
  assert(ss.str() == "12 0 1 0 0 0 0 0 0 0 0 0 0 0 0");

  h.fill(-0.1);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 0 0 0 0 0 0 0 0 0 0 0");

  h.fill(1.1);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 0 0 0 0 0 0 0 0 0 0 1");

  h.fill(0);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 1 0 0 0 0 0 0 0 0 0 1");

  h.fill(0.1);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 1 1 0 0 0 0 0 0 0 0 1");

  h.fill(0.0999);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 2 1 0 0 0 0 0 0 0 0 1");

  h.fill(0.2);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 2 1 1 0 0 0 0 0 0 0 1");

  h.fill(0.9);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 2 1 1 0 0 0 0 0 0 1 1");

  h.fill(0.9999999);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 2 1 1 0 0 0 0 0 0 2 1");

  return 0;
}
