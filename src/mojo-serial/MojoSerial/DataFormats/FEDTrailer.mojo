from std.sys import size_of

from MojoSerial.MojoBridge.DTypes import Typeable, UChar


@fieldwise_init
struct FedtStruct(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var conscheck: UInt32
    var eventsize: UInt32

    @always_inline
    def __init__(out self):
        self.conscheck = 0
        self.eventsize = 0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "FedtStruct"


comptime FedtType = FedtStruct

comptime FED_SLINK_END_MARKER = 0xA

comptime FED_TCTRLID_WIDTH = 0x0000000F
comptime FED_TCTRLID_SHIFT = 28
comptime FED_TCTRLID_MASK = (FED_TCTRLID_WIDTH << FED_TCTRLID_SHIFT)


@always_inline
def FED_TCTRLID_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_TCTRLID_SHIFT) & FED_TCTRLID_WIDTH


comptime FED_EVSZ_WIDTH = 0x00FFFFFF
comptime FED_EVSZ_SHIFT = 0
comptime FED_EVSZ_MASK = (FED_EVSZ_WIDTH << FED_EVSZ_SHIFT)


@always_inline
def FED_EVSZ_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_EVSZ_SHIFT) & FED_EVSZ_WIDTH


comptime FED_CRCS_WIDTH = 0x0000FFFF
comptime FED_CRCS_SHIFT = 16
comptime FED_CRCS_MASK = (FED_CRCS_WIDTH << FED_CRCS_SHIFT)


@always_inline
def FED_CRCS_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_CRCS_SHIFT) & FED_CRCS_WIDTH


comptime FED_STAT_WIDTH = 0x0000000F
comptime FED_STAT_SHIFT = 8
comptime FED_STAT_MASK = (FED_STAT_WIDTH << FED_STAT_SHIFT)


@always_inline
def FED_STAT_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_STAT_SHIFT) & FED_STAT_WIDTH


comptime FED_TTSI_WIDTH = 0x0000000F
comptime FED_TTSI_SHIFT = 4
comptime FED_TTSI_MASK = (FED_TTSI_WIDTH << FED_TTSI_SHIFT)


@always_inline
def FED_TTSI_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_TTSI_SHIFT) & FED_TTSI_WIDTH


comptime FED_MORE_TRAILERS_WIDTH = 0x00000001
comptime FED_MORE_TRAILERS_SHIFT = 3
comptime FED_MORE_TRAILERS_MASK = (
    FED_MORE_TRAILERS_WIDTH << FED_MORE_TRAILERS_SHIFT
)


@always_inline
def FED_MORE_TRAILERS_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_MORE_TRAILERS_SHIFT) & FED_MORE_TRAILERS_WIDTH


comptime FED_CRC_MODIFIED_WIDTH = 0x00000001
comptime FED_CRC_MODIFIED_SHIFT = 2
comptime FED_CRC_MODIFIED_MASK = (FED_CRC_MODIFIED_WIDTH << FED_CRC_MODIFIED_SHIFT)


@always_inline
def FED_CRC_MODIFIED_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_CRC_MODIFIED_SHIFT) & FED_CRC_MODIFIED_WIDTH


comptime FED_SLINK_ERROR_WIDTH = 0x00000001
comptime FED_SLINK_ERROR_SHIFT = 14
comptime FED_SLINK_ERROR_MASK = (FED_SLINK_ERROR_WIDTH << FED_SLINK_ERROR_SHIFT)


@always_inline
def FED_SLINK_ERROR_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_SLINK_ERROR_SHIFT) & FED_SLINK_ERROR_WIDTH


comptime FED_WRONG_FEDID_WIDTH = 0x00000001
comptime FED_WRONG_FEDID_SHIFT = 15
comptime FED_WRONG_FEDID_MASK = (FED_WRONG_FEDID_WIDTH << FED_WRONG_FEDID_SHIFT)


@always_inline
def FED_WRONG_FEDID_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_WRONG_FEDID_SHIFT) & FED_WRONG_FEDID_WIDTH


@fieldwise_init
struct FEDTrailer(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    comptime length: UInt32 = UInt32(size_of[FedtType]())
    # C++ holds `const fedt_struct*` into the raw buffer; this is 8 bytes and
    # read-only, so it is copied instead (doc §18).
    var theTrailer: FedtType

    @always_inline
    def __init__(out self):
        self.theTrailer = FedtType()

    @always_inline
    def fragmentLength(self) -> UInt32:
        return UInt32(FED_EVSZ_EXTRACT(Int(self.theTrailer.eventsize)))

    @always_inline
    def crc(self) -> UInt16:
        return UInt16(FED_CRCS_EXTRACT(Int(self.theTrailer.conscheck)))

    @always_inline
    def evtStatus(self) -> UInt8:
        return UInt8(FED_STAT_EXTRACT(Int(self.theTrailer.conscheck)))

    @always_inline
    def ttsBits(self) -> UInt8:
        return UInt8(FED_TTSI_EXTRACT(Int(self.theTrailer.conscheck)))

    @always_inline
    def moreTrailers(self) -> Bool:
        return FED_MORE_TRAILERS_EXTRACT(Int(self.theTrailer.conscheck)) != 0

    @always_inline
    def crcModified(self) -> Bool:
        return FED_CRC_MODIFIED_EXTRACT(Int(self.theTrailer.conscheck)) != 0

    @always_inline
    def slinkError(self) -> Bool:
        return FED_SLINK_ERROR_EXTRACT(Int(self.theTrailer.conscheck)) != 0

    @always_inline
    def wrongFedId(self) -> Bool:
        return FED_WRONG_FEDID_EXTRACT(Int(self.theTrailer.conscheck)) != 0

    @always_inline
    def check(self) -> Bool:
        return (
            FED_TCTRLID_EXTRACT(Int(self.theTrailer.eventsize))
            == FED_SLINK_END_MARKER
        )

    @always_inline
    def conscheck(self) -> UInt32:
        return self.theTrailer.conscheck

    @staticmethod
    def set(
        trailer: Pointer[mut=True, UChar, _],
        var length: UInt32,
        var crc: UInt16,
        var evtStatus: UInt8,
        var ttsBits: UInt8,
        var moreTrailers: Bool = False,
    ):
        var t = trailer.unsafe_bitcast[FedtType]()

        t[].eventsize = UInt32(
            (FED_SLINK_END_MARKER << FED_TCTRLID_SHIFT)
            | ((Int(length) << FED_EVSZ_SHIFT) & FED_EVSZ_MASK)
        )

        t[].conscheck = UInt32(
            ((Int(crc) << FED_CRCS_SHIFT) & FED_CRCS_MASK)
            | ((Int(evtStatus) << FED_STAT_SHIFT) & FED_STAT_MASK)
            | ((Int(ttsBits) << FED_TTSI_SHIFT) & FED_TTSI_MASK)
        )

        if moreTrailers:
            t[].conscheck |= UInt32(
                FED_MORE_TRAILERS_WIDTH << FED_MORE_TRAILERS_SHIFT
            )

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "FEDTrailer"
