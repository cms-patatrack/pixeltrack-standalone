#ifndef DataFormats_Products_h
#define DataFormats_Products_h

#include <vector>

// The header rootcling parses to build the dictionaries for this backend's
// products.  Which of the types declared here are products is said once, in
// REFLECT_PRODUCTS in the Makefile: that list drives the dictionary, the class
// bindings, the get/emplace dispatch and the name -> type_index table alike.
//
// std::vector<float> is here so that the backend has a product at all: a
// LinkDef with no selection rule is an error, and the reconstruction's own
// products arrive with the reconstruction.

#endif
