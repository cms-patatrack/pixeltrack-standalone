from MojoSerial.CUDADataFormats.GPUClusteringConstants import (
    GPUClusteringConstants,
)
from MojoSerial.DataFormats.SOARotation import SOARotation, SOAFrame
from MojoSerial.Geometry.Phase1PixelTopology import (
    Phase1PixelTopology,
    AverageGeometry,
)
from MojoSerial.MojoBridge.DTypes import Float, Typeable
from MojoSerial.MojoBridge.Vector import Vector

comptime Frame = SOAFrame[DType.float32]
comptime Rotation = SOARotation[DType.float32]


@fieldwise_init
struct CommonParams(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var theThicknessB: Float
    var theThicknessE: Float
    var thePitchX: Float
    var thePitchY: Float

    @always_inline
    def __init__(out self):
        self.theThicknessB = 0.0
        self.theThicknessE = 0.0
        self.thePitchX = 0.0
        self.thePitchY = 0.0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "CommonParams"


@fieldwise_init
struct DetParams(Copyable, Defaultable, Movable, Typeable):
    var isBarrel: Bool
    var isPosZ: Bool
    var layer: UInt16
    var index: UInt16
    var rawId: UInt32

    var shiftX: Float
    var shiftY: Float
    var chargeWidthX: Float
    var chargeWidthY: Float

    var x0: Float
    var y0: Float
    var z0: Float

    var sx: InlineArray[Float, 3]  # errors
    var sy: InlineArray[Float, 3]  # errors

    var frame: Frame

    @always_inline
    def __init__(out self):
        self.isBarrel = False
        self.isPosZ = False
        self.layer = 0
        self.index = 0
        self.rawId = 0

        self.shiftX = 0.0
        self.shiftY = 0.0
        self.chargeWidthX = 0.0
        self.chargeWidthY = 0.0

        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0

        self.sx = InlineArray[Float, 3](fill=0)
        self.sy = InlineArray[Float, 3](fill=0)

        self.frame = Frame()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "DetParams"


@fieldwise_init
struct LayerGeometry(Copyable, Defaultable, Movable, Typeable):
    var layerStart: InlineArray[
        UInt32, Int(Phase1PixelTopology.numberOfLayers + 1)
    ]
    var layer: InlineArray[UInt8, Int(Phase1PixelTopology.layerIndexSize)]

    @always_inline
    def __init__(out self):
        self.layerStart = InlineArray[
            UInt32, Int(Phase1PixelTopology.numberOfLayers + 1)
        ](fill=0)
        self.layer = InlineArray[
            UInt8, Int(Phase1PixelTopology.layerIndexSize)
        ](fill=0)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "LayerGeometry"


# C++ has a ParamsOnGPU handle of four pointers into PixelCPEFast's own fields.
# Its accessors live on PixelCPEFast directly here.


@fieldwise_init
struct ClusParamsT[N: UInt32](Copyable, Defaultable, Movable, Typeable):
    var minRow: InlineArray[UInt32, Int(Self.N)]
    var maxRow: InlineArray[UInt32, Int(Self.N)]
    var minCol: InlineArray[UInt32, Int(Self.N)]
    var maxCol: InlineArray[UInt32, Int(Self.N)]

    var Q_f_X: InlineArray[Int32, Int(Self.N)]
    var Q_l_X: InlineArray[Int32, Int(Self.N)]
    var Q_f_Y: InlineArray[Int32, Int(Self.N)]
    var Q_l_Y: InlineArray[Int32, Int(Self.N)]

    var charge: InlineArray[Int32, Int(Self.N)]

    var xpos: InlineArray[Float, Int(Self.N)]
    var ypos: InlineArray[Float, Int(Self.N)]

    var xerr: InlineArray[Float, Int(Self.N)]
    var yerr: InlineArray[Float, Int(Self.N)]

    var xsize: InlineArray[Int16, Int(Self.N)]  # clipped at 127 if negative is edge
    var ysize: InlineArray[Int16, Int(Self.N)]

    @always_inline
    def __init__(out self):
        self.minRow = InlineArray[UInt32, Int(Self.N)](fill=0)
        self.maxRow = InlineArray[UInt32, Int(Self.N)](fill=0)
        self.minCol = InlineArray[UInt32, Int(Self.N)](fill=0)
        self.maxCol = InlineArray[UInt32, Int(Self.N)](fill=0)

        self.Q_f_X = InlineArray[Int32, Int(Self.N)](fill=0)
        self.Q_l_X = InlineArray[Int32, Int(Self.N)](fill=0)
        self.Q_f_Y = InlineArray[Int32, Int(Self.N)](fill=0)
        self.Q_l_Y = InlineArray[Int32, Int(Self.N)](fill=0)

        self.charge = InlineArray[Int32, Int(Self.N)](fill=0)
        self.xpos = InlineArray[Float, Int(Self.N)](fill=0.0)
        self.ypos = InlineArray[Float, Int(Self.N)](fill=0.0)

        self.xerr = InlineArray[Float, Int(Self.N)](fill=0.0)
        self.yerr = InlineArray[Float, Int(Self.N)](fill=0.0)

        self.xsize = InlineArray[Int16, Int(Self.N)](fill=0)
        self.ysize = InlineArray[Int16, Int(Self.N)](fill=0)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "ClusParamsT[" + String(Self.N) + "]"


comptime MaxHitsInIter: Int32 = GPUClusteringConstants.maxHitsInIter().cast[
    DType.int32
]()
comptime ClusParams = ClusParamsT[MaxHitsInIter.cast[DType.uint32]()]


def computeAnglesFromDet(
    ref detParams: DetParams,
    x: Float,
    y: Float,
    mut cotalpha: Float,
    mut cotbeta: Float,
):
    # x,y local position on det
    var gvx = x - detParams.x0
    var gvy = y - detParams.y0
    var gvz = -1.0 / detParams.z0
    # normalization not required as only ratio used...
    # calculate angles
    cotalpha = gvx * gvz
    cotbeta = gvy * gvz


def correction(
    sizeM1: Int32,
    Q_f: Int32,  # Charge in the first pixel
    Q_l: Int32,  # Charge in the last pixel
    upper_edge_first_pix: UInt16,  # As the name says
    lower_edge_last_pix: UInt16,  # As the name says
    lorentz_shift: Float,  # L-shift at half thickness
    theThickness: Float,  # detector thickness
    cot_angle: Float,  # cot of alpha_ or beta_
    pitch: Float,  # thePitchX or thePitchY
    first_is_big: Bool,  # true if the first is big
    last_is_big: Bool,  # true if the last is big
) -> Float:
    if sizeM1 == 0:  # size 1
        return 0.0

    var W_eff: Float = 0.0
    var simple: Bool = True
    if sizeM1 == 1:  # size 2
        # Width of the clusters minus the edge (first and last) pixels
        # In the note, they are denoted x_F and x_L (and y_F and y_L)
        # C++ keeps this commented out (and as >=): equal edges are valid.
        # debug_assert(lower_edge_last_pix >= upper_edge_first_pix)
        var W_inner = pitch * Float(lower_edge_last_pix - upper_edge_first_pix)

        # Predicted charge width from geometry
        var W_pred = (
            theThickness * cot_angle  # geometric correction (in cm)
            - lorentz_shift
        )  # (in cm) &&& check fpix!

        W_eff = abs(W_pred) - W_inner

        # If the observed charge width is inconsistent with the expectations
        # based on the track, do *not* use W_pred-W_inner.  Instead, replace
        # it with an *average* effective charge width, which is the average
        # length of the edge pixels.
        simple = (W_eff < 0.0) | (
            W_eff > pitch
        )  # this produces "large" regressions for very small numeric differences...

    if simple:
        # Total length of the two edge pixels (first+last)
        var sum_of_edge: Float = 2.0
        if first_is_big:
            sum_of_edge += 1.0
        if last_is_big:
            sum_of_edge += 1.0
        W_eff = (
            pitch * 0.5 * sum_of_edge
        )  # ave. length of edge pixels (first+last) (cm)

    # Finally, compute the position in this projection
    var Qdiff = (Q_l - Q_f).cast[DType.float32]()
    var Qsum = (Q_f + Q_l).cast[DType.float32]()

    # Temporary fix for clusters with both first and last pixel with charge = 0
    if Qsum == 0.0:
        Qsum = 1.0

    return 0.5 * (Qdiff / Qsum) * W_eff


def position(
    ref comParams: CommonParams,
    ref detParams: DetParams,
    mut cp: ClusParams,
    ic: UInt32,
):
    # Upper Right corner of Lower Left pixel -- in measurement frame
    var llx: UInt16 = cp.minRow[ic].cast[DType.uint16]() + 1
    var lly: UInt16 = cp.minCol[ic].cast[DType.uint16]() + 1

    # Lower Left corner of Upper Right pixel -- in measurement frame
    var urx: UInt16 = cp.maxRow[ic].cast[DType.uint16]()
    var ury: UInt16 = cp.maxCol[ic].cast[DType.uint16]()

    var llxl = Phase1PixelTopology.localX(llx)
    var llyl = Phase1PixelTopology.localY(lly)
    var urxl = Phase1PixelTopology.localX(urx)
    var uryl = Phase1PixelTopology.localY(ury)

    var mx = llxl + urxl
    var my = llyl + uryl

    var xsize = Int(urxl) + 2 - Int(llxl)
    var ysize = Int(uryl) + 2 - Int(llyl)
    debug_assert(xsize >= 0)  # 0 if bixpix...
    debug_assert(ysize >= 0)

    if Phase1PixelTopology.isBigPixX(cp.minRow[ic].cast[DType.uint16]()):
        xsize += 1
    if Phase1PixelTopology.isBigPixX(cp.maxRow[ic].cast[DType.uint16]()):
        xsize += 1
    if Phase1PixelTopology.isBigPixY(cp.minCol[ic].cast[DType.uint16]()):
        ysize += 1
    if Phase1PixelTopology.isBigPixY(cp.maxCol[ic].cast[DType.uint16]()):
        ysize += 1

    var unbalanceX = Int(
        8.0
        * abs(Float(cp.Q_f_X[ic] - cp.Q_l_X[ic]))
        / Float(cp.Q_f_X[ic] + cp.Q_l_X[ic])
    )
    var unbalanceY = Int(
        8.0
        * abs(Float(cp.Q_f_Y[ic] - cp.Q_l_Y[ic]))
        / Float(cp.Q_f_Y[ic] + cp.Q_l_Y[ic])
    )
    xsize = 8 * xsize - unbalanceX
    ysize = 8 * ysize - unbalanceY

    cp.xsize[ic] = Int16(min(xsize, 1023))
    cp.ysize[ic] = Int16(min(ysize, 1023))

    if (cp.minRow[ic] == 0) or (
        cp.maxRow[ic]
        == Phase1PixelTopology.lastRowInModule.cast[DType.uint32]()
    ):
        cp.xsize[ic] = -cp.xsize[ic]
    if (cp.minCol[ic] == 0) or (
        cp.maxCol[ic]
        == Phase1PixelTopology.lastColInModule.cast[DType.uint32]()
    ):
        cp.ysize[ic] = -cp.ysize[ic]

    # apply the lorentz offset correction
    var xPos = detParams.shiftX + comParams.thePitchX * (
        0.5 * mx.cast[DType.float32]() + Float(Phase1PixelTopology.xOffset)
    )
    var yPos = detParams.shiftY + comParams.thePitchY * (
        0.5 * my.cast[DType.float32]() + Float(Phase1PixelTopology.yOffset)
    )

    var cotalpha: Float = 0.0
    var cotbeta: Float = 0.0

    computeAnglesFromDet(detParams, xPos, yPos, cotalpha, cotbeta)

    var thickness = (
        comParams.theThicknessB if detParams.isBarrel else comParams.theThicknessE
    )

    var xcorr = correction(
        (cp.maxRow[ic] - cp.minRow[ic]).cast[DType.int32](),
        cp.Q_f_X[ic],
        cp.Q_l_X[ic],
        llxl,
        urxl,
        detParams.chargeWidthX,  # lorentz shift in cm
        thickness,
        cotalpha,
        comParams.thePitchX,
        Phase1PixelTopology.isBigPixX(cp.minRow[ic].cast[DType.uint16]()),
        Phase1PixelTopology.isBigPixX(cp.maxRow[ic].cast[DType.uint16]()),
    )

    var ycorr = correction(
        (cp.maxCol[ic] - cp.minCol[ic]).cast[DType.int32](),
        cp.Q_f_Y[ic],
        cp.Q_l_Y[ic],
        llyl,
        uryl,
        detParams.chargeWidthY,  # lorentz shift in cm
        thickness,
        cotbeta,
        comParams.thePitchY,
        Phase1PixelTopology.isBigPixY(cp.minCol[ic].cast[DType.uint16]()),
        Phase1PixelTopology.isBigPixY(cp.maxCol[ic].cast[DType.uint16]()),
    )

    cp.xpos[ic] = xPos + xcorr
    cp.ypos[ic] = yPos + ycorr


def errorFromSize(
    ref comParams: CommonParams,
    ref detParams: DetParams,
    mut cp: ClusParams,
    var ic: UInt32,
):
    cp.xerr[ic] = 0.0050
    cp.yerr[ic] = 0.0085

    comptime xerr_barrel_l1: InlineArray[Float, 3] = [0.00115, 0.00120, 0.00088]
    comptime xerr_barrel_l1_def: Float = 0.00200  # 0.01030
    comptime yerr_barrel_l1: InlineArray[Float, 9] = [
        0.00375,
        0.00230,
        0.00250,
        0.00250,
        0.00230,
        0.00230,
        0.00210,
        0.00210,
        0.00240,
    ]
    comptime yerr_barrel_l1_def: Float = 0.00210
    comptime xerr_barrel_ln: InlineArray[Float, 3] = [0.00115, 0.00120, 0.00088]
    comptime xerr_barrel_ln_def: Float = 0.00200  # 0.01030
    comptime yerr_barrel_ln: InlineArray[Float, 9] = [
        0.00375,
        0.00230,
        0.00250,
        0.00250,
        0.00230,
        0.00230,
        0.00210,
        0.00210,
        0.00240,
    ]
    comptime yerr_barrel_ln_def: Float = 0.00210
    comptime xerr_endcap: InlineArray[Float, 2] = [0.0020, 0.0020]
    comptime xerr_endcap_def: Float = 0.0020
    comptime yerr_endcap: InlineArray[Float, 1] = [0.00210]
    comptime yerr_endcap_def: Float = 0.00210

    var sx = cp.maxRow[ic] - cp.minRow[ic]
    var sy = cp.maxCol[ic] - cp.minCol[ic]

    # is edgy ?
    var isEdgeX = (cp.minRow[ic] == 0) or (
        cp.maxRow[ic]
        == Phase1PixelTopology.lastRowInModule.cast[DType.uint32]()
    )
    var isEdgeY = (cp.minCol[ic] == 0) or (
        UInt32(cp.maxCol[ic])
        == Phase1PixelTopology.lastColInModule.cast[DType.uint32]()
    )
    # is one and big?
    var isBig1X = (sx == 0) and Phase1PixelTopology.isBigPixX(
        cp.minRow[ic].cast[DType.uint16]()
    )
    var isBig1Y = (sy == 0) and Phase1PixelTopology.isBigPixY(
        cp.minCol[ic].cast[DType.uint16]()
    )

    # The tables stay comptime; `materialize` is what puts one in runtime
    # memory so it can be indexed by the runtime sx/sy.
    if not isEdgeX and not isBig1X:
        if not detParams.isBarrel:
            cp.xerr[ic] = (
                materialize[xerr_endcap]()[Int(sx)] if Int(sx)
                < comptime (len(xerr_endcap)) else xerr_endcap_def
            )
        elif detParams.layer == 1:
            cp.xerr[ic] = (
                materialize[xerr_barrel_l1]()[Int(sx)] if Int(sx)
                < comptime (len(xerr_barrel_l1)) else xerr_barrel_l1_def
            )
        else:
            cp.xerr[ic] = (
                materialize[xerr_barrel_ln]()[Int(sx)] if Int(sx)
                < comptime (len(xerr_barrel_ln)) else xerr_barrel_ln_def
            )

    if not isEdgeY and not isBig1Y:
        if not detParams.isBarrel:
            cp.yerr[ic] = (
                materialize[yerr_endcap]()[Int(sy)] if Int(sy)
                < comptime (len(yerr_endcap)) else yerr_endcap_def
            )
        elif detParams.layer == 1:
            cp.yerr[ic] = (
                materialize[yerr_barrel_l1]()[Int(sy)] if Int(sy)
                < comptime (len(yerr_barrel_l1)) else yerr_barrel_l1_def
            )
        else:
            cp.yerr[ic] = (
                materialize[yerr_barrel_ln]()[Int(sy)] if Int(sy)
                < comptime (len(yerr_barrel_ln)) else yerr_barrel_ln_def
            )


def errorFromDB(
    ref comParams: CommonParams,
    ref detParams: DetParams,
    mut cp: ClusParams,
    ic: UInt32,
):
    # Edge cluster errors
    cp.xerr[ic] = 0.0050
    cp.yerr[ic] = 0.0085

    var sx = cp.maxRow[ic] - cp.minRow[ic]
    var sy = cp.maxCol[ic] - cp.minCol[ic]

    # is edgy ?
    var isEdgeX = (cp.minRow[ic] == 0) or (
        cp.maxRow[ic]
        == Phase1PixelTopology.lastRowInModule.cast[DType.uint32]()
    )
    var isEdgeY = (cp.minCol[ic] == 0) or (
        cp.maxCol[ic]
        == Phase1PixelTopology.lastColInModule.cast[DType.uint32]()
    )
    # is one and big?
    var ix = (
        UInt32(sx == 0)
        + Scalar[DType.bool](
            (sx == 0)
            and Phase1PixelTopology.isBigPixX(
                cp.minRow[ic].cast[DType.uint16]()
            )
        ).cast[DType.uint32]()
    )
    var iy = (
        UInt32(sy == 0)
        + Scalar[DType.bool](
            (sy == 0)
            and Phase1PixelTopology.isBigPixY(
                cp.minCol[ic].cast[DType.uint16]()
            )
        ).cast[DType.uint32]()
    )

    if not isEdgeX:
        cp.xerr[ic] = detParams.sx[Int(ix)]
    if not isEdgeY:
        cp.yerr[ic] = detParams.sy[Int(iy)]
