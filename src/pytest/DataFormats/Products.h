#ifndef DataFormats_Products_h
#define DataFormats_Products_h

#include <vector>

namespace pytest {
  /// The payload of the demo workflow.  The generator's output and both squared
  /// results are this same type, distinguished only by the label of the module
  /// that produced them -- which is what the (type, label) key in
  /// ProductRegistry is for.
  using Floats = std::vector<float>;
}  // namespace pytest

// The header rootcling parses to build the dictionaries for this backend's
// products.  Which of the types declared here are products is said once, in
// REFLECT_PRODUCTS in the Makefile: that list drives the dictionary, the class
// bindings, the get/emplace dispatch and the name -> type_index table alike.
//
// It names std::vector<float> rather than the pytest::Floats alias, because it
// is also the name a Python module asks for and the one ROOT reflected.

#endif
