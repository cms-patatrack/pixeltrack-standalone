from std.collections import Span
from MojoSerial.MojoBridge.DTypes import Float, Typeable


@fieldwise_init
struct SiPixelGainForHLTonGPU_DecodingStructure(
    Copyable, Defaultable, Movable, Typeable,
    TrivialRegisterPassable,
):
    var gain: UInt8
    var ped: UInt8

    @always_inline
    def __init__(out self):
        self.gain = 0
        self.ped = 0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelGainForHLTonGPU_DecodingStructure"


@fieldwise_init
struct SiPixelGainForHLTonGPU(Copyable, Defaultable, Movable, Typeable):
    comptime DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure
    comptime Range = Tuple[UInt32, UInt32]

    # view into SiPixelGainCalibrationForHLTGPU's _gainData; origin untracked
    var v_pedestals: Span[Self.DecodingStructure, ImmUntrackedOrigin]
    var rangeAndCols: InlineArray[Tuple[Self.Range, Int32], 2000]
    var _minPed: Float
    var _maxPed: Float
    var _minGain: Float
    var _maxGain: Float

    var pedPrecision: Float
    var gainPrecision: Float

    var _numberOfRowsAveragedOver: UInt32  # this is 80!!!!
    var _nBinsToUseForEncoding: UInt32
    var _deadFlag: UInt32
    var _noisyFlag: UInt32

    @always_inline
    def __init__(out self):
        self.v_pedestals = Span[
            Self.DecodingStructure, ImmUntrackedOrigin
        ]()
        self.rangeAndCols = InlineArray[Tuple[Self.Range, Int32], 2000](
            fill=Tuple[Self.Range, Int32](Self.Range(0, 0), 0)
        )
        self._minPed = 0.0
        self._maxPed = 0.0
        self._minGain = 0.0
        self._maxGain = 0.0

        self.pedPrecision = 0.0
        self.gainPrecision = 0.0

        self._numberOfRowsAveragedOver = 0
        self._nBinsToUseForEncoding = 0
        self._deadFlag = 0
        self._noisyFlag = 0

    @always_inline
    def getPedAndGain(
        self,
        moduleInd: UInt32,
        col: Int32,
        row: Int32,
        mut isDeadColumn: Bool,
        mut isNoisyColumn: Bool,
    ) -> Tuple[Float, Float]:
        var range = self.rangeAndCols[moduleInd][0]
        var ncols = self.rangeAndCols[moduleInd][1]

        # determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
        var lengthOfColumnData: UInt32 = (
            (range[1].cast[DType.int32]() - range[0].cast[DType.int32]())
            // ncols
        ).cast[DType.uint32]()
        # we always only have two values per column averaged block
        var lengthOfAveragedDataInEachColumn: UInt32 = 2
        var numberOfDataBlocksToSkip = (
            row.cast[DType.uint32]() // self._numberOfRowsAveragedOver
        )
        var offset = (
            range[0]
            + col.cast[DType.uint32]() * lengthOfColumnData
            + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip
        )
        debug_assert(offset < range[1])
        debug_assert(offset < 3088384)
        debug_assert(offset % 2 == 0)

        var lp = self.v_pedestals
        var s = lp[offset // 2]

        isDeadColumn = (s.ped.cast[DType.uint32]() & 0xFF) == (self._deadFlag)
        isNoisyColumn = (s.ped.cast[DType.uint32]() & 0xFF) == (self._noisyFlag)
        return (
            self.decodePed(s.ped.cast[DType.uint32]() & 0xFF),
            self.decodeGain(s.gain.cast[DType.uint32]() & 0xFF),
        )

    @always_inline
    def decodeGain(self, gain: UInt32) -> Float:
        return gain.cast[DType.float32]() * self.gainPrecision + self._minGain

    @always_inline
    def decodePed(self, ped: UInt32) -> Float:
        return ped.cast[DType.float32]() * self.pedPrecision + self._minPed

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelGainForHLTonGPU"
