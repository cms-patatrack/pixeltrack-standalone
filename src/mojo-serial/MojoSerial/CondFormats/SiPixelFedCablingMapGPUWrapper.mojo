from MojoSerial.CondFormats.SiPixelFedCablingMapGPU import (
    SiPixelFedCablingMapGPU,
)
from std.collections import Span

from MojoSerial.MojoBridge.DTypes import UChar, Typeable


# WARNING: THIS STRUCT IS 128-ALIGNED (SiPixelFedCablingMapGPU)
struct SiPixelFedCablingMapGPUWrapper(Defaultable, Movable, Typeable):
    var modToUnpDefault: List[UChar]
    var _hasQuality: Bool
    var cablingMapHost: SiPixelFedCablingMapGPU

    @always_inline
    def __init__(out self):
        self.modToUnpDefault = []
        self._hasQuality = False
        self.cablingMapHost = SiPixelFedCablingMapGPU()

    @always_inline
    def __init__(
        out self,
        var cablingMap: SiPixelFedCablingMapGPU,
        var modToUnp: List[UChar],
    ):
        self.modToUnpDefault = modToUnp^
        self._hasQuality = True
        self.cablingMapHost = cablingMap^

    def hasQuality(self) -> Bool:
        return self._hasQuality

    def getCPUProduct(self) -> ref [self.cablingMapHost] SiPixelFedCablingMapGPU:
        return self.cablingMapHost

    def getModToUnpAll(
        self,
    ) -> Span[UChar, origin_of(self.modToUnpDefault)].Immutable:
        return Span(self.modToUnpDefault)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelFedCablingMapGPUWrapper"
