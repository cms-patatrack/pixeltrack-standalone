from std.sys import size_of

from MojoSerial.CondFormats.PixelCPEforGPU import (
    CommonParams,
    DetParams,
    LayerGeometry,
    AverageGeometry,
)
from MojoSerial.MojoBridge.DTypes import UChar, Typeable
from MojoSerial.CondFormats.PixelGPUDetails import PixelGPUDetails


# C++ is a flat, 128-aligned struct of seven MAX_SIZE (57,600) arrays so it can
# be copied to the device in one block. Held as InlineArrays, moving it by value
# stalls the 1.0 compiler at default optimisation, and serial has no device
# copy, so the arrays live on the heap here. The file layout is unchanged; see
# SiPixelFedCablingMapGPUWrapperESProducer for the field-by-field read.
struct SiPixelFedCablingMapGPU(Defaultable, Movable, Typeable):
    comptime N = Int(PixelGPUDetails.MAX_SIZE)
    var fed: List[UInt32]
    var link: List[UInt32]
    var roc: List[UInt32]
    var RawId: List[UInt32]
    var rocInDet: List[UInt32]
    var moduleId: List[UInt32]
    var badRocs: List[UChar]
    var size: UInt32

    @always_inline
    def __init__(out self):
        self.fed = List[UInt32](length=Self.N, fill=0)
        self.link = List[UInt32](length=Self.N, fill=0)
        self.roc = List[UInt32](length=Self.N, fill=0)
        self.RawId = List[UInt32](length=Self.N, fill=0)
        self.rocInDet = List[UInt32](length=Self.N, fill=0)
        self.moduleId = List[UInt32](length=Self.N, fill=0)
        self.badRocs = List[UChar](length=Self.N, fill=0)
        self.size = 0

    @always_inline
    def __init__(
        out self,
        var fed: List[UInt32],
        var link: List[UInt32],
        var roc: List[UInt32],
        var RawId: List[UInt32],
        var rocInDet: List[UInt32],
        var moduleId: List[UInt32],
        var badRocs: List[UChar],
        size: UInt32,
    ):
        self.fed = fed^
        self.link = link^
        self.roc = roc^
        self.RawId = RawId^
        self.rocInDet = rocInDet^
        self.moduleId = moduleId^
        self.badRocs = badRocs^
        self.size = size

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self.fed = move.fed^
        self.link = move.link^
        self.roc = move.roc^
        self.RawId = move.RawId^
        self.rocInDet = move.rocInDet^
        self.moduleId = move.moduleId^
        self.badRocs = move.badRocs^
        self.size = move.size

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelFedCablingMapGPU"
