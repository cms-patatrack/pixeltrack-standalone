import MojoSerial.plugin_PixelTriplets.CAConstants as CAConstants
from MojoSerial.plugin_PixelTriplets.GPUCACell import GPUCACell
import MojoSerial.plugin_PixelTriplets.gpuPixelDoubletsAlgo as gpuPixelDoubleAlgo
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoSerial.plugin_PixelTriplets.gpuFishbone import fishbone



comptime nPairs: Int = 13 + 2 + 4

comptime layerPairs: InlineArray[UInt8, 2 * nPairs] = [
    0, 1, 0, 4, 0, 7,              # BPIX1 (3)
    1, 2, 1, 4, 1, 7,              # BPIX2 (5)
    4, 5, 7, 8,                    # FPIX1 (8)
    2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  # BPIX3 & FPIX2 (13)
    0, 2, 1, 3,                    # Jumping Barrel (15)
    0, 5, 0, 8,                    # Jumping Forward (BPIX1,FPIX2)
    4, 6, 7, 9                     # Jumping Forward (19)
]

comptime phi0p05: Int16 = 522
comptime phi0p06: Int16 = 626
comptime phi0p07: Int16 = 730

comptime phicuts: InlineArray[Int16, nPairs] = [
    phi0p05,
    phi0p07,
    phi0p07,
    phi0p05,
    phi0p06,
    phi0p06,
    phi0p05,
    phi0p05,
    phi0p06,
    phi0p06,
    phi0p06,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
]

comptime minz: InlineArray[Float32, nPairs] = [
    -20.0,
    0.0,
    -30.0,
    -22.0,
    10.0,
    -30.0,
    -70.0,
    -70.0,
    -22.0,
    15.0,
    -30.0,
    -70.0,
    -70.0,
    -20.0,
    -22.0,
    0.0,
    -30.0,
    -70.0,
    -70.0,
]

comptime maxz: InlineArray[Float32, nPairs] = [
    20.0,
    30.0,
    0.0,
    22.0,
    30.0,
    -10.0,
    70.0,
    70.0,
    22.0,
    30.0,
    -15.0,
    70.0,
    70.0,
    20.0,
    22.0,
    30.0,
    0.0,
    70.0,
    70.0,
]

comptime maxr: InlineArray[Float32, nPairs] = [
    20.0,
    9.0,
    9.0,
    20.0,
    7.0,
    7.0,
    5.0,
    5.0,
    20.0,
    6.0,
    6.0,
    5.0,
    5.0,
    20.0,
    20.0,
    9.0,
    9.0,
    9.0,
    9.0,
]

comptime CellNeighbors = CAConstants.CellNeighbors
comptime CellTracks = CAConstants.CellTracks
comptime CellNeighborsVector = CAConstants.CellNeighborsVector
comptime CellTracksVector = CAConstants.CellTracksVector

def initDoublets(
    isOuterHitOfCell: Span[mut=True, GPUCACell.OuterHitOfCell, _],
    nHits: UInt32,
    mut cellNeighbors: CellNeighborsVector,
    mut cellTracks: CellTracksVector,
):
    var first: UInt32 = 0
    for i in range(first, nHits):
        isOuterHitOfCell[Int(i)].reset()

    if first == 0:
        # C++ also hands `construct` the container buffer; SimpleVector owns
        # its storage here.
        cellNeighbors.construct(Int32(CAConstants.maxNumOfActiveDoublets()))
        cellTracks.construct(Int32(CAConstants.maxNumOfActiveDoublets()))
        var i = cellNeighbors.extend()
        debug_assert(i == 0)
        cellNeighbors[0].reset()
        i = cellTracks.extend()
        debug_assert(i == 0)
        cellTracks[0].reset()


comptime getDoubletsFromHistoMaxBlockSize: Int = 64
comptime getDoubletsFromHistoMinBlocksPerMP: Int = 16
#TO-DO port this 
##ifdef __CUDACC__
# __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)
##endif
def getDoubletsFromHisto(
    cells: Span[mut=True, GPUCACell, _],
    mut nCells: UInt32,
    mut cellNeighbors: CellNeighborsVector,
    mut cellTracks: CellTracksVector,
    hh: TrackingRecHit2DHeterogeneous,
    isOuterHitOfCell: Span[mut=True, GPUCACell.OuterHitOfCell, _],
    nActualPairs: Int,
    ideal_cond: Bool,
    doClusterCut: Bool,
    doZ0Cut: Bool,
    doPtCut: Bool,
    maxNumOfDoublets: UInt32,
):
    # `comptime assert` is only legal in a function body, so this guard on the
    # module-level tables lives with their consumer.
    comptime assert nPairs <= Int(CAConstants.maxNumberOfLayerPairs())

    # the tables are comptime, and a Span needs runtime storage to borrow
    var layerPairsRT = materialize[layerPairs]()
    var phicutsRT = materialize[phicuts]()
    var minzRT = materialize[minz]()
    var maxzRT = materialize[maxz]()
    var maxrRT = materialize[maxr]()

    gpuPixelDoubleAlgo.doubletsFromHisto(
        Span(layerPairsRT),
        UInt32(nActualPairs),
        cells,
        nCells,
        cellNeighbors,
        cellTracks,
        hh,
        isOuterHitOfCell,
        Span(phicutsRT),
        Span(minzRT),
        Span(maxzRT),
        Span(maxrRT),
        ideal_cond,
        doClusterCut,
        doZ0Cut,
        doPtCut,
        maxNumOfDoublets,
    )
    
