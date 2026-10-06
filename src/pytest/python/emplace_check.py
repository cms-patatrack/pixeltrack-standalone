"""Reads back what emplace_demo built, so the round trip is checked in Python.

The C++ Printer checks the aggregate; this checks the vector, which no C++
module in this backend consumes.
"""

import numpy as np

import edm_core  # built into the executable

FLOATS = "std::vector<float>"


class EmplaceCheck:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.values = registry.consumes(FLOATS, config["input"])

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        values = event.get(self.values)
        expected = np.arange(values.size(), dtype=np.float32) ** 2
        got = np.asarray(values.data)
        if not np.array_equal(got, expected):
            raise RuntimeError(f"emplaced vector is {got}, expected {expected}")
        print(f"EmplaceCheck Event {event.eventID} stream {event.streamID}: "
              f"{values.size()} values, {got.tolist()}, as emplaced", flush=True)


def create(config: dict, registry: edm_core.ProductRegistry) -> EmplaceCheck:
    return EmplaceCheck(config, registry)
