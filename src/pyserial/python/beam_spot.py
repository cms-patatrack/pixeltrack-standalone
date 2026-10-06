"""The beam spot, copied from the EventSetup into the Event, in Python.

A port of plugin-BeamSpotProducer/BeamSpotToPOD, which is one line of C++:

    iEvent.emplace(bsPutToken_, iSetup.get<BeamSpotPOD>());

It is here to be measured rather than to be interesting.  The vertex finder
prices Python compute over a few hundred tracks; this module computes nothing
and moves eleven floats, so what it costs per event is very nearly the fixed
price of the crossing itself -- entering the interpreter, building the two
argument objects, resolving the product.  The two numbers bracket the cost of
any other module.

It is also the smallest module that needs the EventSetup, which is why the
EventSetup is reachable from Python at all.
"""

import edm_core  # built into the executable

BEAM_SPOT = "BeamSpotPOD"

FIELDS = (
    "x",
    "y",
    "z",
    "sigmaZ",
    "beamWidthX",
    "beamWidthY",
    "dxdz",
    "dydz",
    "emittanceX",
    "emittanceY",
    "betaStar",
)


class BeamSpotProducer:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.beamSpot = registry.produces(BEAM_SPOT)

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        source = eventSetup.get(BEAM_SPOT)
        target = event.allocate(self.beamSpot)
        for field in FIELDS:
            setattr(target, field, getattr(source, field))


def create(config: dict, registry: edm_core.ProductRegistry) -> BeamSpotProducer:
    return BeamSpotProducer(config, registry)
