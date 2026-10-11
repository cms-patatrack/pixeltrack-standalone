from std.sys import size_of

from MojoSerial.MojoBridge.DTypes import Typeable, UChar


@fieldwise_init
struct FedhStruct(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var sourceid: UInt32
    var eventid: UInt32

    @always_inline
    def __init__(out self):
        self.sourceid = 0
        self.eventid = 0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "FedhStruct"


comptime FedhType = FedhStruct

comptime FED_SLINK_START_MARKER = 0x5

comptime FED_HCTRLID_WIDTH = 0x0000000F
comptime FED_HCTRLID_SHIFT = 28
comptime FED_HCTRLID_MASK = (FED_HCTRLID_WIDTH << FED_HCTRLID_SHIFT)


@always_inline
def FED_HCTRLID_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_HCTRLID_SHIFT) & FED_HCTRLID_WIDTH


comptime FED_EVTY_WIDTH = 0x0000000F
comptime FED_EVTY_SHIFT = 24
comptime FED_EVTY_MASK = (FED_EVTY_WIDTH << FED_EVTY_SHIFT)


@always_inline
def FED_EVTY_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_EVTY_SHIFT) & FED_EVTY_WIDTH


comptime FED_LVL1_WIDTH = 0x00FFFFFF
comptime FED_LVL1_SHIFT = 0
comptime FED_LVL1_MASK = (FED_LVL1_WIDTH << FED_LVL1_SHIFT)


@always_inline
def FED_LVL1_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_LVL1_SHIFT) & FED_LVL1_WIDTH


comptime FED_BXID_WIDTH = 0x00000FFF
comptime FED_BXID_SHIFT = 20
comptime FED_BXID_MASK = (FED_BXID_WIDTH << FED_BXID_SHIFT)


@always_inline
def FED_BXID_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_BXID_SHIFT) & FED_BXID_WIDTH


comptime FED_SOID_WIDTH = 0x00000FFF
comptime FED_SOID_SHIFT = 8
comptime FED_SOID_MASK = (FED_SOID_WIDTH << FED_SOID_SHIFT)


@always_inline
def FED_SOID_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_SOID_SHIFT) & FED_SOID_WIDTH


comptime FED_VERSION_WIDTH = 0x0000000F
comptime FED_VERSION_SHIFT = 4
comptime FED_VERSION_MASK = (FED_VERSION_WIDTH << FED_VERSION_SHIFT)


@always_inline
def FED_VERSION_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_VERSION_SHIFT) & FED_VERSION_WIDTH


comptime FED_MORE_HEADERS_WIDTH = 0x00000001
comptime FED_MORE_HEADERS_SHIFT = 3
comptime FED_MORE_HEADERS_MASK = (FED_MORE_HEADERS_WIDTH << FED_MORE_HEADERS_SHIFT)


@always_inline
def FED_MORE_HEADERS_EXTRACT(a: Int) -> Int:
    return ((a) >> FED_MORE_HEADERS_SHIFT) & FED_MORE_HEADERS_WIDTH


@fieldwise_init
struct FEDHeader(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    comptime length: UInt32 = UInt32(size_of[FedhType]())
    # C++ holds `const fedh_struct*` into the raw buffer; this is 8 bytes and
    # read-only, so it is copied instead (doc §18).
    var theHeader: FedhType

    @always_inline
    def __init__(out self):
        self.theHeader = FedhType()

    @always_inline
    def triggerType(self) -> UInt8:
        return UInt8(FED_EVTY_EXTRACT(Int(self.theHeader.eventid)))

    @always_inline
    def lvl1ID(self) -> UInt32:
        return UInt32(FED_LVL1_EXTRACT(Int(self.theHeader.eventid)))

    @always_inline
    def bxID(self) -> UInt16:
        return UInt16(FED_BXID_EXTRACT(Int(self.theHeader.sourceid)))

    @always_inline
    def sourceID(self) -> UInt16:
        return UInt16(FED_SOID_EXTRACT(Int(self.theHeader.sourceid)))

    @always_inline
    def version(self) -> UInt8:
        return UInt8(FED_VERSION_EXTRACT(Int(self.theHeader.sourceid)))

    @always_inline
    def moreHeaders(self) -> Bool:
        return FED_MORE_HEADERS_EXTRACT(Int(self.theHeader.sourceid)) != 0

    @always_inline
    def check(self) -> Bool:
        return (
            FED_HCTRLID_EXTRACT(Int(self.theHeader.eventid))
            == FED_SLINK_START_MARKER
        )

    @staticmethod
    def set(
        header: Pointer[mut=True, UChar, _],
        triggerType: UInt8,
        lvl1ID: UInt32,
        bxID: UInt16,
        sourceID: UInt16,
        version: UInt8 = 0,
        moreHeaders: Bool = False,
    ):
        var h = header.unsafe_bitcast[FedhType]()
        h[].eventid = UInt32(
            (FED_SLINK_START_MARKER << FED_HCTRLID_SHIFT)
            | ((Int(triggerType) << FED_EVTY_SHIFT) & FED_EVTY_MASK)
            | ((Int(lvl1ID) << FED_LVL1_SHIFT) & FED_LVL1_MASK)
        )
        h[].sourceid = UInt32(
            ((Int(bxID) << FED_BXID_SHIFT) & FED_BXID_MASK)
            | ((Int(sourceID) << FED_SOID_SHIFT) & FED_SOID_MASK)
            | ((Int(version) << FED_VERSION_SHIFT) & FED_VERSION_MASK)
        )
        if moreHeaders:
            h[].sourceid |= UInt32(
                FED_MORE_HEADERS_WIDTH << FED_MORE_HEADERS_SHIFT
            )

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "FEDHeader"
