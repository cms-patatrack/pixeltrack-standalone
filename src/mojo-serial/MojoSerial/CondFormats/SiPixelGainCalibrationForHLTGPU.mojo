from std.collections import Span

from MojoSerial.CondFormats.SiPixelGainForHLTonGPU import SiPixelGainForHLTonGPU
from MojoSerial.MojoBridge.DTypes import Char, Typeable


struct SiPixelGainCalibrationForHLTGPU(Defaultable, Movable, Typeable):
    # Inline, not OwnedPointer: 24 kB is well under the size where boxing costs
    # anything at compile time (measured), and getCPUProduct() returns a borrow,
    # so the box only bought an allocation and an indirection. v_pedestals below
    # points into _gainData's heap buffer, which keeps its address when this
    # struct moves.
    var _gainForHLTonHost: SiPixelGainForHLTonGPU
    var _gainData: List[Char]

    @always_inline
    def __init__(out self):
        self._gainForHLTonHost = SiPixelGainForHLTonGPU()
        self._gainData = []

    @always_inline
    def __init__(
        out self, gain: SiPixelGainForHLTonGPU, var gainData: List[Char]
    ):
        self._gainData = gainData^
        self._gainForHLTonHost = gain.copy()
        self._gainForHLTonHost.v_pedestals = rebind[
            Span[SiPixelGainForHLTonGPU.DecodingStructure, ImmUntrackedOrigin]
        ](
            Span[SiPixelGainForHLTonGPU.DecodingStructure, _](
                unsafe_ptr=self._gainData.unsafe_ptr().bitcast[
                    SiPixelGainForHLTonGPU.DecodingStructure
                ](),
                length=len(self._gainData) // 2,
            )
        )

    @always_inline
    def getCPUProduct(
        self,
    ) -> ref [self._gainForHLTonHost] SiPixelGainForHLTonGPU:
        return self._gainForHLTonHost

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelGainCalibrationForHLTGPU"
