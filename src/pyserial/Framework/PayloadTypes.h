#ifndef PayloadTypes_h
#define PayloadTypes_h

#include <string>
#include <typeindex>
#include <vector>

namespace edm {
  // A Python module names a product type with a string, and the registry keys
  // on a std::type_index; these bridge the two.  Both are *defined* in the file
  // tools/generate_bindings.C writes, from the product list the Makefile hands
  // it -- the same list that drives rootcling and the class bindings -- so the
  // set of payload types is written once and cannot drift.

  /// Every payload type name, in the order the build listed them.
  std::vector<std::string> const& payloadTypeNames();

  /// The std::type_index a payload name stands for.  Throws, listing what is
  /// known, if the name is not a payload type.
  std::type_index payloadTypeIndex(std::string const& name);
}  // namespace edm

#endif
