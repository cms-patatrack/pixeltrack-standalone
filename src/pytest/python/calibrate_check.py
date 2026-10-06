"""Reads both Samples products back, so the get() side is checked in Python too.

calibrate.py writes; this reads.  Everything here comes out of event.get(),
which hands back the product a C++ module published -- the columns as numpy
views over its storage, the nested view by reference, the scalar by value --
and the last line copies a class product into the Event with put(), which is
the only one of the three ways to publish that takes a finished object.
"""

import numpy as np

import edm_core  # built into the executable

SAMPLES = "pytest::Samples"
COMPARISON = "pytest::Comparison"
COUNT = "int"

COLUMNS = ("valueSpan", "scaledSpan", "channelSpan")


class CalibrateCheck:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.lhs = registry.consumes(SAMPLES, config["lhs"])
        self.rhs = registry.consumes(SAMPLES, config["rhs"])
        self.summary = registry.consumes(COMPARISON, config["summary"])
        self.count = registry.consumes(COUNT, config.get("count", ""))
        self.copy = registry.produces(COMPARISON)

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        lhs = event.get(self.lhs)
        rhs = event.get(self.rhs)

        if lhs.size() != rhs.size():
            raise RuntimeError(f"{lhs.size()} samples against {rhs.size()}")
        if lhs.size() > edm_core.Samples.kMaxSamples:
            raise RuntimeError(f"{lhs.size()} samples, more than kMaxSamples")

        left, right = lhs.view(), rhs.view()
        if left.filled != right.filled or left.size() != right.size():
            raise RuntimeError(f"views differ: {left.filled}/{left.size()} against "
                               f"{right.filled}/{right.size()}")
        for column in COLUMNS:
            a = np.asarray(getattr(left, column))
            b = np.asarray(getattr(right, column))
            if a.shape != (lhs.size(),):
                raise RuntimeError(f"{column} is {a.shape}, expected {(lhs.size(),)}")
            if not np.array_equal(a, b):
                raise RuntimeError(f"{column} differs in {int(np.count_nonzero(a != b))} of {a.size}")

        # A scalar product crosses by value; a class product comes back as a
        # reference to what is in the Event.
        if event.get(self.count) != rhs.size():
            raise RuntimeError(f"count is {event.get(self.count)}, {rhs.size()} samples")

        summary = event.get(self.summary)
        if not summary.agree() or summary.size != lhs.size():
            raise RuntimeError(f"{summary.mismatches} mismatches in {summary.size} values")

        # put() copies a finished product into the Event: the only one of
        # allocate / put / emplace that takes an object built elsewhere.
        event.put(self.copy, summary)

        print(f"CalibrateCheck Event {event.eventID} stream {event.streamID}: "
              f"{lhs.size()} samples, {len(COLUMNS)} columns identical, "
              f"largest deviation {summary.maxDeviation}", flush=True)


def create(config: dict, registry: edm_core.ProductRegistry) -> CalibrateCheck:
    return CalibrateCheck(config, registry)
