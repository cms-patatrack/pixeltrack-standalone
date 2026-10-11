from std.pathlib import Path

from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.MojoBridge.DTypes import Typeable

trait ESProducer(Copyable, Defaultable, Movable, Typeable):
    def __init__(out self, var path: Path):
        ...

    def produce(mut self, mut eventSetup: EventSetup):
        ...
