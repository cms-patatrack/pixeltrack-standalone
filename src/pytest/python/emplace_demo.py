"""Builds products from Python values, to exercise Event.emplace().

The three operations the Event offers differ in who owns the storage and who
fills it, and this module uses the one that has to construct:

    event.allocate(token)          storage in the Event, filled in place
    event.put(token, value)        a copy of a product you already have
    event.emplace(token, *values)  built here from Python values, moved in

emplace() is what turns Python values into a C++ product, so it is the only
one that takes a list where a std::vector is wanted -- put() needs a
std::vector already, since copying is all it does.

What can be built is decided per product by the binding generator, from the
dictionary: a scalar takes one value, a std::vector of scalars takes any
sequence, and an aggregate whose members are all scalars takes one value per
member, in declaration order.
"""

import edm_core  # built into the executable

FLOATS = "std::vector<float>"
COMPARISON = "pytest::Comparison"

N = 8


class EmplaceDemo:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.values = registry.produces(FLOATS)
        self.summary = registry.produces(COMPARISON)

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        # a std::vector<float> built from a Python list
        squares = [float(i * i) for i in range(N)]
        event.emplace(self.values, squares)

        # an aggregate built from its members, in declaration order:
        # size, mismatches, maxDeviation, tolerance
        event.emplace(self.summary, N, 0, 0.0, 0.0)


def create(config: dict, registry: edm_core.ProductRegistry) -> EmplaceDemo:
    return EmplaceDemo(config, registry)
