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
    mut cellNeighbors: CellNeighborsVector,
    mut cellTracks: CellTracksVector,
):
    # C++ also hands `construct` the container buffer; SimpleVector owns
    # its storage here.
    cellNeighbors.construct_for_overwrite(Int32(CAConstants.maxNumOfActiveDoublets()))
    cellTracks.construct_for_overwrite(Int32(CAConstants.maxNumOfActiveDoublets()))
    var i = cellNeighbors.extend()
    debug_assert(i == 0)
    cellNeighbors[0].reset()
    i = cellTracks.extend()
    debug_assert(i == 0)
    cellTracks[0].reset()


# Fill the compact storage of the cells whose outer hit is each hit, from the cells found by getDoubletsFromHisto.
# The cells are added in increasing order, up to maxCellsPerHit() per hit, so the result is the same as calling
# OuterHitOfCell.push_back() for each cell as soon as it is created.
def fillOuterHitOfCell(
    cells: Span[GPUCACell, _],
    nCells: UInt32,
    nHits: UInt32,
    mut isOuterHitOfCell: CAConstants.OuterHitOfCellContainer,
):
    comptime maxCellsPerHit = CAConstants.maxCellsPerHit()
    var nh = Int(nHits)
    var nc = Int(nCells)
    var offsets = List[UInt32](unsafe_uninit_length=nh + 1)
    var cursor = List[UInt32](unsafe_uninit_length=nh)
    var outerCells = List[UInt32](unsafe_uninit_length=max(nc, 1))

    # count the cells for each hit, in offsets[hit + 1]
    for i in range(nh + 1):
        offsets[i] = 0
    for c in range(nc):
        var k = Int(cells[c].get_outer_hit_id()) + 1
        offsets[k] += UInt32(offsets[k] < maxCellsPerHit)
    # convert the counts to offsets
    for i in range(nh):
        offsets[i + 1] += offsets[i]
        cursor[i] = offsets[i]
    # fill the cells
    for c in range(nc):
        var hit = Int(cells[c].get_outer_hit_id())
        if cursor[hit] < offsets[hit + 1]:
            outerCells[Int(cursor[hit])] = UInt32(c)
            cursor[hit] += 1

    isOuterHitOfCell.offsets = offsets^
    isOuterHitOfCell.cells = outerCells^


def getDoubletsFromHisto(
    cells: Span[mut=True, GPUCACell, _],
    mut nCells: UInt32,
    mut cellNeighbors: CellNeighborsVector,
    mut cellTracks: CellTracksVector,
    hh: TrackingRecHit2DHeterogeneous,
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
    
