from std.pathlib import Path
from std.sys.info import size_of
from std.memory import memcpy

from MojoSerial.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoSerial.Framework.ESProducer import ESProducer
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.MojoBridge.DTypes import Char, Typeable, UChar
from MojoSerial.MojoBridge.File import read_obj

@fieldwise_init
struct BeamSpotESProducer(ESProducer):
    var _data: Path

    @always_inline
    def __init__(out self):
        self._data = Path("")

    @always_inline
    def produce(mut self, mut eventSetup: EventSetup):
        try:
            with open(self._data / "beamspot.bin", "r") as file:
                var bs = read_obj[BeamSpotPOD](file)
                eventSetup.put[BeamSpotPOD](
                    bs
                )
        except e:
            print(
                (
                    "Error during loading data in"
                    " BeamSpotESProducer:"
                ),
                e,
            )

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "BeamSpotESProducer"
