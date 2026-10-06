"""Builds a Samples product through every binding the generator emits.

pyserial's Python modules reach the Event in a handful of ways, and between
them they use one of everything tools/generate_bindings.C can produce.  Nothing
in pyserial can be turned into a test -- it needs the pixel detector, its
conditions and its raw data -- so this module stands in for it: same shapes,
arithmetic simple enough that numpy and C++ agree bit for bit.

What it exercises, in the order it appears below:

    eventSetup.get(type)            a class product read by type, no token
    edm_core.Calibration.kChannels  a static constant, from the class
    calibration.channels()          a method with no arguments
    calibration.channel(i)          a method taking an argument
    channel.gain.slope()            a class member, and a method on it
    channel.weights                 a fixed-size array as a numpy view
    channel.index, channel.enabled  scalar members
    calibration.view()              a nested class, returned by reference
    view.slopeSpan                  a std::span column as a numpy view
    registry.consumes(type, "")     the unique product of a type
    event.allocate(token, n, ...)   a constructor that wires the product
    samples.view()                  a nested class, returned by reference
    view.valueSpan[:] = ...         writing a column in place, no copy
    view.filled = n                 writing a scalar member through that view
    samples.applyCalibration()      a method with a side effect on the product
    event.put(token, n)             a scalar product, by value

CalibrateCxx builds the same product in C++ and SamplesCompare checks the two
column by column, so a binding that hands back a copy, or a view of the wrong
length, fails the job rather than quietly reporting a plausible number.
"""

import numpy as np

import edm_core  # built into the executable

FLOATS = "std::vector<float>"
SAMPLES = "pytest::Samples"
CALIBRATION = "pytest::Calibration"
COUNT = "int"


class Calibrate:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        # An empty label means "the only product of that type": this job has
        # one Floats producer, so there is nothing to name.
        self.input = registry.consumes(FLOATS, config.get("input", ""))
        self.output = registry.produces(SAMPLES)
        self.count = registry.produces(COUNT)
        self.checked = False

    # ------------------------------------------------------------ conditions
    @staticmethod
    def _check_conditions(calibration: edm_core.Calibration) -> None:
        """Reads the calibration both ways and checks it against its literals.

        Once per stream: these are conditions, and D20 measured a bound scalar
        read at ~35 ns, which is a Python-level loop nobody wants per event.
        """
        channels = calibration.channels()
        if channels != edm_core.Calibration.kChannels:
            raise RuntimeError(f"channels() is {channels}, kChannels is {edm_core.Calibration.kChannels}")

        for i in range(channels):
            channel = calibration.channel(i)
            gain = channel.gain
            if gain.slope() != 1.0 + 0.25 * i or gain.offset() != -0.5 * i:
                raise RuntimeError(f"channel {i} gain is ({gain.slope()}, {gain.offset()})")
            weights = np.asarray(channel.weights)
            if weights.shape != (3,) or not np.array_equal(weights, [0.5 * i + w for w in range(3)]):
                raise RuntimeError(f"channel {i} weights are {weights}")
            if channel.index != i or channel.enabled != (i != 3):
                raise RuntimeError(f"channel {i} is index={channel.index} enabled={channel.enabled}")

        # The same numbers as columns, which is how the arithmetic below reads
        # them: one numpy array instead of eight bound attribute reads.
        columns = calibration.view()
        slope = np.asarray(columns.slopeSpan)
        offset = np.asarray(columns.offsetSpan)
        enabled = np.asarray(columns.enabledSpan)
        if slope.shape != (channels,) or offset.shape != (channels,) or enabled.shape != (channels,):
            raise RuntimeError(f"calibration columns are {slope.shape}, {offset.shape}, {enabled.shape}")
        for i in range(channels):
            channel = calibration.channel(i)
            if slope[i] != channel.gain.slope() or offset[i] != channel.gain.offset():
                raise RuntimeError(f"channel {i} disagrees with the columns")
            if bool(enabled[i]) != channel.enabled:
                raise RuntimeError(f"channel {i} enabled disagrees with the column")

    def _check_errors(self, event: edm_core.Event) -> None:
        """The two ways of putting a product that cannot work, and have to say
        so rather than corrupt the Event."""
        try:
            event.allocate(self.count)
        except TypeError:
            pass
        else:
            raise RuntimeError("allocate() of a scalar product should have failed")

        try:
            event.put(self.output, None)
        except TypeError:
            pass
        else:
            raise RuntimeError("put() of a product that cannot be copied should have failed")

    # --------------------------------------------------------------- produce
    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        calibration = eventSetup.get(CALIBRATION)
        if not self.checked:
            self._check_conditions(calibration)
            self._check_errors(event)
            self.checked = True

        values = np.asarray(event.get(self.input).data)
        n = int(values.size)
        channels = (np.arange(n, dtype=np.int32) % calibration.channels()).astype(np.int32)

        # Samples has no default constructor worth using: it is built around
        # the conditions and the channel column, and what comes back is empty
        # storage in the Event to fill.
        samples = event.allocate(self.output, n, calibration, channels)
        view = samples.view()
        np.asarray(view.valueSpan)[:] = values
        view.filled = n
        samples.applyCalibration()

        # applyCalibration() ran in C++; the same arithmetic here, from the
        # calibration columns, has to give the same bits.  It is a subtraction
        # and a multiplication, never a multiply-add, so no FMA can appear on
        # one side and not the other.
        columns = calibration.view()
        slope = np.asarray(columns.slopeSpan)[channels]
        offset = np.asarray(columns.offsetSpan)[channels]
        enabled = np.asarray(columns.enabledSpan)[channels].astype(bool)
        expected = np.where(enabled, (values - offset) * slope, np.float32(0.0))
        scaled = np.asarray(view.scaledSpan)
        if not np.array_equal(scaled, expected):
            worst = int(np.argmax(np.abs(scaled - expected)))
            raise RuntimeError(f"applyCalibration() disagrees at {worst}: "
                               f"{scaled[worst]} against {expected[worst]}")

        # A scalar product: no storage to fill, so it is copied in by value.
        event.put(self.count, n)


def create(config: dict, registry: edm_core.ProductRegistry) -> Calibrate:
    return Calibrate(config, registry)
