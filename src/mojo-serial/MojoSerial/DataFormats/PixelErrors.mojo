from MojoSerial.DataFormats.SiPixelRawDataError import SiPixelRawDataError
from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct PixelErrorCompact(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var raw_id: UInt32
    var word: UInt32
    var error_type: UInt8
    var fed_id: UInt8

    @always_inline
    def __init__(out self):
        self.raw_id = 0
        self.word = 0
        self.error_type = 0
        self.fed_id = 0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "PixelErrorCompact"


comptime PixelFormatterErrors = Dict[UInt, List[SiPixelRawDataError]]
