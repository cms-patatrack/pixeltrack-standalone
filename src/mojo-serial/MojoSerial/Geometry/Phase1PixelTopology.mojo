from MojoSerial.MojoBridge.DTypes import Float, Typeable
struct Phase1PixelTopology:
    comptime numRowsInRoc: UInt16 = 80
    comptime numColsInRoc: UInt16 = 52
    comptime lastRowInRoc: UInt16 = Self.numRowsInRoc - 1
    comptime lastColInRoc: UInt16 = Self.numColsInRoc - 1

    comptime numRowsInModule: UInt16 = 2 * Self.numRowsInRoc
    comptime numColsInModule: UInt16 = 8 * Self.numColsInRoc
    comptime lastRowInModule: UInt16 = Self.numRowsInModule - 1
    comptime lastColInModule: UInt16 = Self.numColsInModule - 1

    comptime xOffset: Int16 = -81
    comptime yOffset: Int16 = -54 * 4

    comptime numPixsInModule: UInt32 = Self.numRowsInModule.cast[
        DType.uint32
    ]() * Self.numColsInModule.cast[DType.uint32]()

    comptime numberOfModules: UInt32 = 1856
    # Int, not UInt32: used as an InlineArray length below, and `Int(<UInt32
    # comptime>)` does not fold to a literal in parameter position, so a list
    # literal cannot match the deduced size.
    comptime numberOfLayers: Int = 10
    comptime layerStart: InlineArray[UInt32, Self.numberOfLayers + 1] = [
        0,
        96,
        320,
        672,  # barrel
        1184,
        1296,
        1408,  # positive endcap
        1520,
        1632,
        1744,  # negative endcap
        Self.numberOfModules,
    ]

    comptime layerName: InlineArray[StaticString, Self.numberOfLayers] = [
        "BL1",
        "BL2",
        "BL3",
        "BL4",  # barrel
        "E+1",
        "E+2",
        "E+3",  # positive endcap
        "E-1",
        "E-2",
        "E-3",  # negative endcap
    ]

    comptime numberOfModulesInBarrel: UInt32 = 1184
    comptime numberOfLaddersInBarrel: UInt32 = Self.numberOfModulesInBarrel / 8

    @staticmethod
    def findMaxModuleStride() -> UInt32:
        # layerStart is indexed against the runtime `n`, so materialize it once
        var ls = materialize[Self.layerStart]()
        var go = True
        var n: UInt32 = 2
        while go:
            for i in range(1, 11):
                if ls[i] % n != 0:
                    go = False
                    break
            if not go:
                break
            n *= 2
        return n // 2

    comptime maxModuleStride = Self.findMaxModuleStride()

    @staticmethod
    def findLayer[detId: UInt32]() -> UInt8:
        comptime for i in range(11):
            comptime bound = Self.layerStart[i + 1]
            if detId < bound:
                return UInt8(i)
        return 11

    @staticmethod
    def findLayerFromCompact[detId: UInt32]() -> UInt8:
        comptime _detId = detId * Self.maxModuleStride

        comptime for i in range(11):
            comptime bound = Self.layerStart[i + 1]
            if _detId < bound:
                return UInt8(i)
        return 11

    comptime layerIndexSize: UInt32 = Self.numberOfModules / Self.maxModuleStride

    # C++ builds this with a generic map-to-array helper taking a function
    # parameter; that shape no longer binds in 1.0, and it had one caller.
    @staticmethod
    def _build_layer_table() -> InlineArray[UInt8, Int(Self.layerIndexSize)]:
        var arr = InlineArray[UInt8, Int(Self.layerIndexSize)](fill=0)
        comptime for i in range(Int(Self.layerIndexSize)):
            arr[i] = Self.findLayerFromCompact[UInt32(i)]()
        return arr^

    comptime layer: InlineArray[
        UInt8, Int(Self.layerIndexSize)
    ] = Self._build_layer_table()

    @staticmethod
    def validateLayerIndex() -> Bool:
        var layer = materialize[Self.layer]()
        var ls = materialize[Self.layerStart]()
        var stride = Int(Self.maxModuleStride)
        var res = True
        for i in range(Int(Self.numberOfModules)):
            var j = i // stride
            var l = Int(layer[j])
            res &= l < 10
            res &= UInt32(i) >= ls[l]
            res &= UInt32(i) < ls[l + 1]
        return res

    comptime __d = debug_assert(
        Self.validateLayerIndex(), "layer from detIndex algo is buggy"
    )

    @always_inline
    @staticmethod
    def divu52(var n: UInt16) -> UInt16:
        """
        This is for the ROC n<512 (upgrade 1024).
        """
        n = n >> 2
        var q = (n >> 1) + (n >> 4)
        q = q + (q >> 4) + (q >> 5)
        q = q >> 3
        var r = n - q * 13
        return q + ((r + 3) >> 4)

    @staticmethod
    @always_inline
    def isEdgeX(px: UInt16) -> Bool:
        return px == 0 or px == Self.lastRowInModule

    @staticmethod
    @always_inline
    def isEdgeY(py: UInt16) -> Bool:
        return py == 0 or py == Self.lastColInModule

    @staticmethod
    @always_inline
    def toRocX(px: UInt16) -> UInt16:
        return px if px < Self.numRowsInRoc else px - Self.numRowsInRoc

    @staticmethod
    @always_inline
    def toRocY(py: UInt16) -> UInt16:
        return py - 52 * Self.divu52(py)

    @staticmethod
    @always_inline
    def isBigPixX(px: UInt16) -> Bool:
        return px == 79 or px == 80

    @staticmethod
    @always_inline
    def isBigPixY(py: UInt16) -> Bool:
        var ly = Self.toRocY(py)
        return ly == 0 or ly == Self.lastColInRoc

    @staticmethod
    @always_inline
    def localX(px: UInt16) -> UInt16:
        var shift: UInt16 = 0
        if px > Self.lastRowInRoc:
            shift += 1
        if px > Self.numRowsInRoc:
            shift += 1
        return px + shift

    @staticmethod
    @always_inline
    def localY(py: UInt16) -> UInt16:
        var roc = Self.divu52(py)
        var shift = 2 * roc
        var yInRoc = py - 52 * roc
        if yInRoc > 0:
            shift += 1
        return py + shift


@fieldwise_init
struct AverageGeometry(Defaultable, Movable, Typeable):
    comptime numberOfLaddersInBarrel = Phase1PixelTopology.numberOfLaddersInBarrel
    var ladderZ: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderX: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderY: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderR: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderMinZ: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderMaxZ: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var endCapZ: InlineArray[Float, 2]

    def __init__(out self):
        self.ladderZ = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderX = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderY = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderR = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderMinZ = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderMaxZ = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.endCapZ = InlineArray[Float, 2](fill=0)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "AverageGeometry"
