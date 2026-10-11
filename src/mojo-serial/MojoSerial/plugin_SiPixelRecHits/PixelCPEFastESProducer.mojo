from std.pathlib import Path

from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from MojoSerial.Framework.ESProducer import ESProducer
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.MojoBridge.DTypes import Typeable


struct PixelCPEFastESProducer(ESProducer):
    var _data: Path

    @always_inline
    def __init__(out self):
        self._data = Path("")

    @always_inline
    def __init__(out self, var path: Path):
        self._data = path^

    @always_inline
    def produce(mut self, mut eventSetup: EventSetup):
        try:
            eventSetup.put[PixelCPEFast](
                PixelCPEFast(self._data / "cpefast.bin")
            )
        except e:
            print(
                "Error during loading data in PixelCPEFastESProducer:",
                e,
            )

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "PixelCPEFastESProducer"
