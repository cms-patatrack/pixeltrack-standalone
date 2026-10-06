"""Python implementation of the squaring stage.

PythonProducer imports this file and calls create() once per stream, with the
module's configuration section as a dict and the ProductRegistry the module is
being constructed against -- the same object, and the same two calls, a C++
module gets in its own constructor.  What they return are tokens, and a token
is the only way to reach a product: declaring a dependency and being able to
read it are the same act.

No float is copied anywhere here, and no binding was written by hand: the
std::vector<float> product is bound from its ROOT dictionary, and `.data` is a
numpy view of the vector's own storage.  The output vector is constructed
inside the Event, sized, and then filled in place.

np.square is a single correctly-rounded IEEE multiply per element, which is
what SquareCxx does too, so the two agree bit for bit and Compare can run with
tolerance = 0.
"""

import numpy as np

import edm_core  # built into the executable

FLOATS = "std::vector<float>"
ES_INT = "int"


class Square:
    """Same contract as a C++ producer: a produce(event) method."""

    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.input = registry.consumes(FLOATS, config["input"])
        self.output = registry.produces(FLOATS)

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        # The EventSetup, read the way a C++ module reads it: by type, with no
        # token.  IntESProducer puts a bare int there, which is the case a
        # dictionary cannot describe -- there is no TClass for a fundamental
        # type -- so it crosses by value through nanobind's own caster, exactly
        # as a scalar *member* of a product already does.  Checking it here is
        # what exercises that path.
        if eventSetup.get(ES_INT) != 42:
            raise RuntimeError(f"EventSetup int is {eventSetup.get(ES_INT)}, expected 42")

        src = event.get(self.input)
        dst = event.allocate(self.output)
        dst.resize(src.size())
        np.square(src.data, out=dst.data)        # writes into C++ memory


def create(config: dict, registry: edm_core.ProductRegistry) -> Square:
    """Factory looked up by PythonProducer."""
    return Square(config, registry)
