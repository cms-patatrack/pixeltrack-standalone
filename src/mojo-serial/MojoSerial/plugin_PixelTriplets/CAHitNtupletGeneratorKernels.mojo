import std.math as math

from std.atomic import Atomic, Ordering
from std.sys import is_defined


import MojoSerial.plugin_PixelTriplets.CAConstants as CAConstants
import MojoSerial.plugin_PixelTriplets.gpuPixelDoublets as gpuPixelDoublets
from MojoSerial.plugin_PixelTriplets.GPUCACell import GPUCACell
from MojoSerial.CUDACore.AtomicPairCounter import AtomicPairCounter
from MojoSerial.CUDACore.CUDACompat import CUDAStreamType
from MojoSerial.CUDACore.HistoContainer import (
    finalizeBulk,
    launchFinalize,
    launchZero,
)
from MojoSerial.CUDADataFormats.GPUClusteringConstants import (
    GPUClusteringConstants,
)
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrack as pixelTrack,
    TrackQuality as trackQuality,
)
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)

comptime HitToTuple = CAConstants.HitToTuple
comptime TupleMultiplicity = CAConstants.TupleMultiplicity
comptime Quality = pixelTrack.Quality
comptime TkSoA = pixelTrack.TrackSoA
comptime HitContainer = pixelTrack.HitContainer
comptime Stream = CUDAStreamType


@fieldwise_init
struct Counters(Defaultable, Movable):
    var nEvents: UInt64
    var nHits: UInt64
    var nCells: UInt64
    var nTuples: UInt64
    var nFitTracks: UInt64
    var nGoodTracks: UInt64
    var nUsedHits: UInt64
    var nDupHits: UInt64
    var nKilledCells: UInt64
    var nEmptyCells: UInt64
    var nZeroTrackCells: UInt64

    def __init__(out self):
        self.nEvents = 0
        self.nHits = 0
        self.nCells = 0
        self.nTuples = 0
        self.nFitTracks = 0
        self.nGoodTracks = 0
        self.nUsedHits = 0
        self.nDupHits = 0
        self.nKilledCells = 0
        self.nEmptyCells = 0
        self.nZeroTrackCells = 0


@fieldwise_init
struct QualityCutsRegion(Copyable, ImplicitlyCopyable, Movable):
    var maxTip: Float32  # cm
    var minPt: Float32  # GeV
    var maxZip: Float32  # cm


@fieldwise_init
struct QualityCuts(Copyable, ImplicitlyCopyable, Movable):
    # chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    # SIMD, not InlineArray: QualityCuts is ImplicitlyCopyable and InlineArray
    # is not; indexing reads the same
    var chi2Coeff: SIMD[DType.float32, 4]
    var chi2MaxPt: Float32  # GeV
    var chi2Scale: Float32

    var triplet: QualityCutsRegion
    var quadruplet: QualityCutsRegion


struct Params(Copyable, ImplicitlyCopyable, Movable):
    var onGPU: Bool
    var minHitsPerNtuplet: UInt32
    var maxNumberOfDoublets: UInt32
    var useRiemannFit: Bool
    var fit5as4: Bool
    var includeJumpingForwardDoublets: Bool
    var earlyFishbone: Bool
    var lateFishbone: Bool
    var idealConditions: Bool
    var doStats: Bool
    var doClusterCut: Bool
    var doZ0Cut: Bool
    var doPtCut: Bool
    var ptmin: Float32
    var CAThetaCutBarrel: Float32
    var CAThetaCutForward: Float32
    var hardCurvCut: Float32
    var dcaCutInnerTriplet: Float32
    var dcaCutOuterTriplet: Float32
    var cuts: QualityCuts

    def __init__(
        out self,
        onGPU: Bool,
        minHitsPerNtuplet: UInt32,
        maxNumberOfDoublets: UInt32,
        useRiemannFit: Bool,
        fit5as4: Bool,
        includeJumpingForwardDoublets: Bool,
        earlyFishbone: Bool,
        lateFishbone: Bool,
        idealConditions: Bool,
        doStats: Bool,
        doClusterCut: Bool,
        doZ0Cut: Bool,
        doPtCut: Bool,
        ptmin: Float32,
        CAThetaCutBarrel: Float32,
        CAThetaCutForward: Float32,
        hardCurvCut: Float32,
        dcaCutInnerTriplet: Float32,
        dcaCutOuterTriplet: Float32,
        cuts: QualityCuts = QualityCuts(
            # polynomial coefficients for the pT-dependent chi2 cut
            chi2Coeff=SIMD[DType.float32, 4](
                0.68177776, 0.74609577, -0.08035491, 0.00315399
            ),
            # max pT used to determine the chi2 cut
            chi2MaxPt=10.0,
            # chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
            chi2Scale=30.0,
            # regional cuts for triplets
            triplet=QualityCutsRegion(maxTip=0.3, minPt=0.5, maxZip=12.0),
            # regional cuts for quadruplets
            quadruplet=QualityCutsRegion(maxTip=0.5, minPt=0.3, maxZip=12.0),
        ),
    ):
        self.onGPU = onGPU
        self.minHitsPerNtuplet = minHitsPerNtuplet
        self.maxNumberOfDoublets = maxNumberOfDoublets
        self.useRiemannFit = useRiemannFit
        self.fit5as4 = fit5as4
        self.includeJumpingForwardDoublets = includeJumpingForwardDoublets
        self.earlyFishbone = earlyFishbone
        self.lateFishbone = lateFishbone
        self.idealConditions = idealConditions
        self.doStats = doStats
        self.doClusterCut = doClusterCut
        self.doZ0Cut = doZ0Cut
        self.doPtCut = doPtCut
        self.ptmin = ptmin
        self.CAThetaCutBarrel = CAThetaCutBarrel
        self.CAThetaCutForward = CAThetaCutForward
        self.hardCurvCut = hardCurvCut
        self.dcaCutInnerTriplet = dcaCutInnerTriplet
        self.dcaCutOuterTriplet = dcaCutOuterTriplet
        self.cuts = cuts


# --- Low-level kernel helpers (C++: CAHitNtupletGeneratorKernelsImpl.h) ---
#
# These were originally in a separate file, CAHitNtupletGeneratorKernelsImpl.mojo,
# imported by this one (for CAHitNtupletGeneratorKernelsCPU's own struct
# methods to call) while it in turn imported Counters/QualityCuts back from
# this file -- a circular import. That shape compiled fine for simple cases
# (verified with an isolated two-file repro), but broke here with a genuine
# Mojo compiler bug: `Counters`/`QualityCuts` ended up as two non-unified
# type identities depending on which file started compilation, producing
# baffling "UnsafePointer[Counters] cannot convert to UnsafePointer[Counters]"
# errors. Merging both files' contents into this single one sidesteps the
# cycle entirely -- these functions still are ordinary free functions (they
# were free/namespace-level functions in the C++ original too, not member
# functions), just no longer split across two mutually-importing files.
def Kernel_checkOverflows(
    foundNtuplets: HitContainer,
    tupleMultiplicity: TupleMultiplicity,
    apc: AtomicPairCounter,
    cells: Span[GPUCACell, _],  # __restrict__ dropped
    nCells: UInt32,  # uint32_t const*
    cellNeighbors: gpuPixelDoublets.CellNeighborsVector,
    cellTracks: gpuPixelDoublets.CellTracksVector,
    isOuterHitOfCell: CAConstants.OuterHitOfCellContainer,
    nHits: UInt32,
    maxNumberOfDoublets: UInt32,
    mut counters: Counters,
):
    var first: UInt32 = 0

    ref c = counters
    if first == 0 :
        Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
            UnsafePointer(to=c.nEvents),
            UInt64(1),
        )
        Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
            UnsafePointer(to=c.nHits),
            UInt64(nHits),
        )
        Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
            UnsafePointer(to=c.nCells),
            UInt64(nCells),
        )
        Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
            UnsafePointer(to=c.nTuples),
            UInt64(apc.get()[1]),
        )
        Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
            UnsafePointer(to=c.nFitTracks),
            UInt64(tupleMultiplicity.size()),
        )

    comptime if is_defined["NTUPLE_DEBUG"]():
        if first == 0:
            print(
                "number of found cells",
                nCells,
                "found tuples",
                apc.get()[1],
                "with total hits",
                apc.get()[0],
                "out of",
                nHits,
            )
            if apc.get()[1] < CAConstants.maxNumberOfQuadruplets():
                debug_assert(
                    foundNtuplets.size(apc.get()[1]) == 0,
                    "Expected size 0",
                )
                debug_assert(
                    foundNtuplets.size() == apc.get()[0],
                    "Size mismatch",
                )
    comptime if is_defined["NTUPLE_DEBUG"]():
        var nBins = Int(foundNtuplets.nbins())
        for idx in range(Int(first), nBins, 1):
            var idx_u = UInt32(idx)
            if foundNtuplets.size(idx_u) > 5:
                print("ERROR", idx, ",", foundNtuplets.size(idx_u))
            debug_assert(foundNtuplets.size(idx_u) < 6)

            var ih = foundNtuplets.begin(idx_u)
            var end = foundNtuplets.end(idx_u)
            while ih != end:
                debug_assert(UInt32(ih[]) < nHits)
                ih += 1
    if first == 0:
        if apc.get()[1] >= CAConstants.maxNumberOfQuadruplets():
            print("Tuples overflow")
        if nCells >= maxNumberOfDoublets:
            print("Cells overflow")
        if cellNeighbors.full():
            print("cellNeighbors overflow")
        if cellTracks.full():
            print("cellTracks overflow")

    var idx: Int = Int(first)
    var nt = Int(nCells)

    while idx  < nt:
        ref thisCell = cells[idx]
        if (thisCell.outerNeighbors(cellNeighbors).full()) : #++tooManyNeighbors[thisCell.theLayerPairId]
          print("OuterNeighbors overflow ",idx , "in \n", thisCell.theLayerPairId)
        if (thisCell.tracks(cellTracks).full()) : #++tooManyTracks[thisCell.theLayerPairId]
          print("Tracks overflow " , idx , " in \n", thisCell.theLayerPairId)
        if (thisCell.theDoubletId < 0):
          Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
              UnsafePointer(to=c.nKilledCells),
              UInt64(1),
          )
        if (thisCell.theUsed == 0):
          Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
              UnsafePointer(to=c.nEmptyCells),
              UInt64(1),
          )
        if (thisCell.tracks(cellTracks).empty()):
          Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
              UnsafePointer(to=c.nZeroTrackCells),
              UInt64(1),
          )
        idx+=1


    idx = Int(first)
    nt = Int(nHits)

    while idx < nt :
        if isOuterHitOfCell.full(idx):
            print("OuterHitOfCell ovberflow " , idx , "\n")

        idx += 1


def kernel_fishboneCleaner(
    cells: Span[GPUCACell, _],
    nCells: UInt32,
    cellTracks: gpuPixelDoublets.CellTracksVector,
    quality: Span[mut=True, Quality, _],
):
    comptime bad = trackQuality.bad
    var nt = Int(nCells)
    for idx in range(0, nt, 1):
        ref thisCell = cells[idx]
        if thisCell.theDoubletId >= 0:
            continue

        ref trk = thisCell.tracks(cellTracks)
        for it in trk.span():
            quality[Int(it)] = bad


def kernel_earlyDuplicateRemover(
    cells: Span[GPUCACell, _],
    nCells: UInt32,
    cellTracks: gpuPixelDoublets.CellTracksVector,
    foundNtuplets: HitContainer,
    quality: Span[mut=True, Quality, _],
):
    comptime dup = trackQuality.dup

    var nt = Int(nCells)
    for idx in range(0, nt, 1):
        ref thisCell = cells[idx]

        if len(thisCell.tracks(cellTracks)) < 2:
            continue

        var maxNh: UInt32 = 0

        ref trk = thisCell.tracks(cellTracks)
        for it in trk.span():
            var nh = foundNtuplets.size(UInt32(it))
            maxNh = max(nh, maxNh)

        for it in trk.span():
            if foundNtuplets.size(UInt32(it)) != maxNh:
                quality[Int(it)] = dup


def kernel_fastDuplicateRemover(
    cells: Span[GPUCACell, _],
    nCells: UInt32,
    cellTracks: gpuPixelDoublets.CellTracksVector,
    # C++ also takes `foundNtuplets` but never reads it; it is a field of
    # `tracks`, so passing both would alias.
    mut tracks: TkSoA,
):
    var bad = trackQuality.bad
    var dup = trackQuality.dup
    var loose = trackQuality.loose

    var nt = Int(nCells)
    for idx in range(0, nt, 1):
        ref thisCell = cells[idx]
        if len(thisCell.tracks(cellTracks)) < 2:
            continue

        var mc: Float32 = 10000.0
        var im: UInt16 = 60000

        # `tracks` is passed in rather than captured: nested closures capture
        # by copy, and TkSoA is not Copyable.
        def score(tracks: TkSoA, it: UInt16) -> Float32:
            return abs(tracks.tip(Int32(it))) # tip
            # or chi2
        #find min socre
        ref trk = thisCell.tracks(cellTracks)
        for it in trk.span():
            if tracks.quality(Int(it)) == loose and score(tracks, it) < mc:
                mc = score(tracks, it)
                im = it
        #mark all other duplicates
        for it in trk.span():
            if tracks.quality(Int(it)) != bad and it != im:
                tracks.quality(Int(it)) = dup # no race:  simple assignment of the same constant


def kernel_connect(
            mut apc1: AtomicPairCounter,
            mut apc2: AtomicPairCounter,  # just to zero them
            hh: GPUCACell.Hits,
            cells: Span[mut=True, GPUCACell, _],
            nCells: UInt32,
            mut cellNeighbors: gpuPixelDoublets.CellNeighborsVector,
            isOuterHitOfCell: CAConstants.OuterHitOfCellContainer,
            hardCurvCut: Float32,
            ptmin: Float32,
            CAThetaCutBarrel: Float32,
            CAThetaCutForward: Float32,
            dcaCutInnerTriplet: Float32,
            dcaCutOuterTriplet: Float32
        ):
    var firstCellIndex = 0 + 0 * 1
    var first: UInt32 = 0
    var stride = 1

    if(0 == (firstCellIndex + Int(first))):
        apc1 = AtomicPairCounter(0)
        apc2 = AtomicPairCounter(0)

    var idx : Int = firstCellIndex
    var nt = Int(nCells)
    #loop on outer cells
    while idx < nt:
        var cellIndex  =idx
        # C++ binds `thisCell`/`oc` as references into the same array and
        # mutates both; two live refs into one Span do not compose here, so
        # each access indexes afresh.
        var innerHitIdx = Int(cells[idx].get_inner_hit_id())
        var numberOfPossibleNeighbors : Int = isOuterHitOfCell.size(innerHitIdx)
        var vb = isOuterHitOfCell.begin(innerHitIdx)
        ref vi = isOuterHitOfCell.cells

        var last_bpix1_detIndex: UInt32 = 96
        var last_barrel_detIndex: UInt32 = 1184
        var ri = cells[idx].get_inner_r(hh)
        var zi = cells[idx].get_inner_z(hh)

        var ro = cells[idx].get_outer_r(hh)
        var zo = cells[idx].get_outer_z(hh)
        var isBarrel = cells[idx].get_inner_detIndex(hh) < Float32(last_barrel_detIndex)
        #loop on inner cells
        for j in range(Int(first), numberOfPossibleNeighbors, stride):
            var otherCell = Int(vi[vb + j])

            var r1 = cells[otherCell].get_inner_r(hh)
            var z1 = cells[otherCell].get_inner_z(hh)

            var aligned : Bool = GPUCACell.areAlignedRZ(
                r1,
                z1,
                ri,
                zi,
                ro,
                zo,
                ptmin,
                CAThetaCutBarrel if isBarrel else  CAThetaCutForward
            )
            var dcaCut = (
                dcaCutInnerTriplet
                if cells[otherCell].get_inner_detIndex(hh)
                < Float32(last_bpix1_detIndex)
                else dcaCutOuterTriplet
            )
            if aligned and cells[idx].dcaCut(
                hh,
                cells[otherCell],
                dcaCut,
                hardCurvCut,
            ):
                cells[otherCell].addOuterNeighbor(
                    UInt32(cellIndex), cellNeighbors
                )
                cells[idx].theUsed |= 1
                cells[otherCell].theUsed |= 1
        idx += 1

def kernel_find_ntuplets(
    hh : GPUCACell.Hits,
    cells : Span[mut=True, GPUCACell, _],
    nCells : UInt32,
    cellNeighbors : gpuPixelDoublets.CellNeighborsVector,
    mut cellTracks : gpuPixelDoublets.CellTracksVector,
    mut foundNtuplets : HitContainer,
    mut apc : AtomicPairCounter,
    quality : Span[mut=True, Quality, _],
    minHitsPerNtuplet : UInt32
    ):
    var nt = Int(nCells)
    for idx in range(0, nt, 1):
        # a copy, not a ref: find_ntuplets only reads it, and holding a borrow
        # of `cells` here would conflict with passing the array itself
        var thisCell = cells[idx].copy()
        if thisCell.theDoubletId < 0:
            continue

        var pid = Int(thisCell.theLayerPairId)
        var doit: Bool = (pid < 3) if minHitsPerNtuplet > 3 else (pid < 8 or pid > 12)
        if doit:
            var stack = GPUCACell.TmpTuple()
            stack.reset()
            thisCell.find_ntuplets[6](
                hh,
                cells,
                cellNeighbors,
                cellTracks,
                foundNtuplets,
                apc,
                quality,
                stack,
                minHitsPerNtuplet,
                pid < 3,
            )
            debug_assert(stack.empty())

def kernel_mark_used(
    hh: GPUCACell.Hits,
    cells: Span[mut=True, GPUCACell, _],
    nCells: UInt32,
    cellTracks: gpuPixelDoublets.CellTracksVector,
):
    var nt = Int(nCells)
    for idx in range(0, nt, 1):
        if not cells[idx].tracks(cellTracks).empty():
            cells[idx].theUsed |= 2

def kernel_countMultiplicity(  foundNtuplets : HitContainer,
                              quality : Span[Quality, _],
                              mut tupleMultiplicity : CAConstants.TupleMultiplicity):
    var nt = Int(foundNtuplets.nbins())
    for it in range(0, nt, 1):
        var it_u = UInt32(it)
        var nhits = foundNtuplets.size(it_u)
        if nhits < 3:
            continue
        if quality[it] == trackQuality.dup:
            continue
        debug_assert(quality[it] == trackQuality.bad)
        if nhits > 5:
            print("wrong mult", it, nhits)
        debug_assert(nhits < 8)
        tupleMultiplicity.countDirect(nhits)

def kernel_fillMultiplicity(foundNtuplets : HitContainer,
                              quality : Span[Quality, _],
                              mut tupleMultiplicity : CAConstants.TupleMultiplicity):
    var nt = Int(foundNtuplets.nbins())
    for it in range(0, nt, 1):
        var it_u = UInt32(it)
        var nhits = foundNtuplets.size(it_u)
        if nhits < 3:
            continue
        if quality[it] == trackQuality.dup:
            continue
        debug_assert(quality[it] == trackQuality.bad)
        if nhits > 5:
            print("wrong mult", it, nhits)
        debug_assert(nhits < 8)
        tupleMultiplicity.fillDirect(
            nhits,
            it_u.cast[DType.uint16](),
        )

def kernel_classifyTracks(
    mut tracks: TkSoA,
    cuts: QualityCuts,
):
    # C++ passes tuples/tracks/quality as three raw pointers; here tuples and
    # quality are fields *of* tracks, so taking them separately aliases. One
    # borrow, derived internally.
    var nt = Int(tracks.hitIndices.nbins())
    for it in range(0, nt, 1):
        var it_u = UInt32(it)
        var it_i = Int32(it)
        var nhits = tracks.hitIndices.size(it_u)
        if nhits == 0:
            break # guard

        #id duplicate : not even fit
        if tracks.quality(it) == trackQuality.dup:
            continue

        debug_assert(tracks.quality(it) == trackQuality.bad)

        #mark doublets as bad
        if nhits < 3:
            continue

        #if the fit has my invalid parameters , mark it as bad
        var isNaN : Bool = False
        for i in range(0, 5, 1):
            isNaN = isNaN or Bool(math.isnan(tracks.stateAtBS.state[it_i][i, 0]))

        if isNaN:
            comptime if is_defined["NTUPLE_DEBUG"]():
                print(
                    "NaN in fit",
                    it,
                    "size",
                    nhits,
                    "chi2",
                    tracks.chi2[it],
                )
            continue
        # compute a pT-dependent chi2 cut
        # default parameters:
        #   - chi2MaxPt = 10 GeV
        #   - chi2Coeff = { 0.68177776, 0.74609577, -0.08035491, 0.00315399 }
        #   - chi2Scale = 30 for broken line fit, 45 for Riemann fit
        # (see CAHitNtupletGeneratorGPU.cc)
        var pt: Float32 = min(tracks.pt[it], cuts.chi2MaxPt)
        var chi2Cut: Float32 = cuts.chi2Scale * (
            cuts.chi2Coeff[0]
            + pt
            * (cuts.chi2Coeff[1] + pt * (cuts.chi2Coeff[2] + pt * cuts.chi2Coeff[3]))
        )
        # above number were for Quads not normalized so for the time being just multiple by ndof for Quads  (triplets to be understood)
        if 3.0 * tracks.chi2[it] >= chi2Cut:
            comptime if is_defined["NTUPLE_DEBUG"]():
                print(
                    "Bad fit",
                    it,
                    "size",
                    nhits,
                    "pt",
                    tracks.pt[it],
                    "eta",
                    tracks.eta[it],
                    "chi2",
                    3.0 * tracks.chi2[it],
                )
            continue
        # impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
        # default cuts:
        #   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
        #   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
        # (see CAHitNtupletGeneratorGPU.cc)
        var region = cuts.quadruplet if nhits > 3 else cuts.triplet
        var tip = tracks.tip(it_i)
        var zip = tracks.zip(it_i)
        var isOk: Bool = (
            abs(tip) < region.maxTip
            and tracks.pt[it] > region.minPt
            and abs(zip) < region.maxZip
        )

        if isOk:
            tracks.quality(it) = trackQuality.loose

def kernel_doStatsForTracks(tuples  : HitContainer ,
                           quality : Span[Quality, _] ,
                           mut counters : Counters ):
    var nt = Int(tuples.nbins())
    for idx in range(0, nt, 1):
        var idx_u = UInt32(idx)
        if tuples.size(idx_u) == 0:
            break # guard
        if quality[idx] != trackQuality.loose:
            continue
        Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
            UnsafePointer(to=(counters.nGoodTracks)),
            UInt64(1),
        )


def kernel_countHitInTracks(tuples  : HitContainer ,
                           quality : Span[Quality, _] ,
                           mut hitToTuple : HitToTuple ):
    var nt = Int(tuples.nbins())
    for idx in range(0, nt, 1):
        var idx_u = UInt32(idx)
        if tuples.size(idx_u) == 0:
            break # guard
        if quality[idx] != trackQuality.loose:
            continue
        var h  = tuples.begin(idx_u)
        var end = tuples.end(idx_u)
        while h != end:
            hitToTuple.countDirect(UInt32(h[]))
            h += 1

def kernel_fillHitInTracks(tuples  : HitContainer ,
                           quality : Span[Quality, _] ,
                           mut hitToTuple : HitToTuple ):
    var nt = Int(tuples.nbins())

    for idx in range(0, nt, 1):
        var idx_u = UInt32(idx)
        if tuples.size(idx_u) == 0:
            break #guard
        if quality[idx] != trackQuality.loose:
            continue
        var h  = tuples.begin(idx_u)
        var end = tuples.end(idx_u)
        while h != end:
            hitToTuple.fillDirect(
                UInt32(h[]),
                idx_u.cast[DType.uint16](),
            )
            h += 1

def kernel_fillHitDetIndices(tuples  : HitContainer ,
                           hh : TrackingRecHit2DHeterogeneous ,
                           mut hitDetIndices : HitContainer ):
    # copy offsets
    var total_bins = Int(tuples.totbins())
    for idx in range(0, total_bins, 1):
        hitDetIndices.off[idx] = tuples.off[idx]
    # fill hit indices
    var nhits = hh.nHits()
    var total_size = Int(tuples.size())
    for idx in range(0, total_size, 1):
        debug_assert(UInt32(tuples.bins[idx]) < nhits)
        hitDetIndices.bins[idx] = hh.detectorIndex(Int(tuples.bins[idx]))

def kernel_doStatsForHitInTracks(hitToTuple: HitToTuple , mut counters :  Counters):
    ref c = counters
    var nt = Int(hitToTuple.nbins())
    for idx in range(0, nt, 1):
        var idx_u = UInt32(idx)
        if hitToTuple.size(idx_u) == 0:
            continue # SHALL NOT BE break
        Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
            UnsafePointer(to=c.nUsedHits),
            UInt64(1),
        )
        if hitToTuple.size(idx_u) > 1 :
            Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](
                UnsafePointer(to=c.nDupHits),
                UInt64(1),
            )

def kernel_tripletCleaner(hh : TrackingRecHit2DHeterogeneous , mut tracks : TkSoA , hitToTuple : HitToTuple):
    # foundNtuplets and quality are fields of tracks; taking them separately
    # aliases, so they are derived from the one borrow
    ref foundNtuplets = tracks.hitIndices
    var bad  = trackQuality.bad
    var dup  = trackQuality.dup

    # loop over hits
    var nt = Int(hitToTuple.nbins())
    for idx in range(0, nt, 1):
        var idx_u = UInt32(idx)
        if hitToTuple.size(idx_u) < 2:
            continue

        var mc: Float32 = 10000.0
        var im: UInt16 = 60000
        var maxNh: UInt32 = 0

        # find maxNh
        var it = hitToTuple.begin(idx_u)
        var it_end = hitToTuple.end(idx_u)
        while it != it_end:
            var nh: UInt32 = foundNtuplets.size(UInt32(it[]))
            maxNh = max(nh, maxNh)
            it += 1

        # kill all tracks shorter than maxHn (only triplets???)
        it = hitToTuple.begin(idx_u)
        while it != it_end:
            var nh: UInt32 = foundNtuplets.size(UInt32(it[]))
            if maxNh != nh:
                tracks.quality(Int(it[])) = dup
            it += 1

        if maxNh > 3:
            continue

        # for triplets choose best tip!
        var ip = hitToTuple.begin(idx_u)
        var ip_end = hitToTuple.end(idx_u)
        while ip != ip_end:
            var it_val = ip[]
            var it_i = Int32(it_val)
            var q = tracks.quality(Int(it_val))
            if q != bad and abs(tracks.tip(it_i)) < mc:
                mc = abs(tracks.tip(it_i))
                im = it_val
            ip += 1

        ip = hitToTuple.begin(idx_u)
        while ip != ip_end:
            var it_val = ip[]
            if tracks.quality(Int(it_val)) != bad and it_val != im:
                tracks.quality(Int(it_val)) = dup # no race:  simple assignment of the same constant
            ip += 1


def kernel_print_found_ntuplets(hh : TrackingRecHit2DHeterogeneous , tracks : TkSoA , hitToTuple : HitToTuple ,  maxPrint : UInt32  ,  iev : Int):
    ref foundNtuplets = tracks.hitIndices
    var i: Int = 0
    while i < Int(min(maxPrint, foundNtuplets.nbins())):
        var i_u = UInt32(i)
        var nh = foundNtuplets.size(i_u)
        if nh < 3:
            i += 1
            continue
        print(
            "TK:",
            10000 * iev + i,
            Int(tracks.quality(i)),
            nh,
            tracks.charge(Int32(i)),
            tracks.pt[i],
            tracks.eta[i],
            tracks.phi(Int32(i)),
            tracks.tip(Int32(i)),
            tracks.zip(Int32(i)),
            tracks.chi2[i],
            (foundNtuplets.begin(i_u))[],
            (foundNtuplets.begin(i_u) + 1)[],
            (foundNtuplets.begin(i_u) + 2)[],
            Int((foundNtuplets.begin(i_u) + 3)[]) if nh > 3 else -1,
            Int((foundNtuplets.begin(i_u) + 4)[]) if nh > 4 else -1,
        )
        i += 1

def kernel_printCounters(
    counters: Counters,
):
    ref c = counters
    print(
        "||Counters | nEvents | nHits | nCells | nTuples | nFitTracks | nGoodTracks | nUsedHits | nDupHits | "
        "nKilledCells | nEmptyCells | nZeroTrackCells ||"
    )
    print(
        "Counters Raw",
        c.nEvents,
        c.nHits,
        c.nCells,
        c.nTuples,
        c.nGoodTracks,
        c.nFitTracks,
        c.nUsedHits,
        c.nDupHits,
        c.nKilledCells,
        c.nEmptyCells,
        c.nZeroTrackCells,
    )
    var events = Float64(c.nEvents)
    var cells = Float64(c.nCells)
    print(
        "Counters Norm",
        c.nEvents,
        Float64(c.nHits) / events,
        Float64(c.nCells) / events,
        Float64(c.nTuples) / events,
        Float64(c.nFitTracks) / events,
        Float64(c.nGoodTracks) / events,
        Float64(c.nUsedHits) / events,
        Float64(c.nDupHits) / events,
        Float64(c.nKilledCells) / events,
        Float64(c.nEmptyCells) / cells,
        Float64(c.nZeroTrackCells) / cells,
    )


# --- Main struct (CPU kernel driver) ---
#
# The C++ original is a class template `CAHitNtupletGeneratorKernels<TTraits>`,
# parameterized so the same code can target CPU or GPU backends via a traits
# type providing `unique_ptr<T>`/allocation helpers. Mojo has no traits-with-
# fields, and this port only ever targets the CPU (serial) backend, so this
# collapses to one concrete struct -- matching
# `CAHitNtupletGeneratorKernelsCPU = CAHitNtupletGeneratorKernels<CPUTraits>`,
# the only instantiation the serial backend ever used anyway.
#
# Field ownership: single-object `unique_ptr<T>` fields become plain inline
# fields. They were `OwnedPointer[T]` while `HitToTuple`/`TupleMultiplicity`
# were ~384 kB/~48 kB inline structs; now that `HistoContainer` is heap-backed
# both are 56 B, so the box is pure overhead. `unique_ptr<T[]>` fields -- dynamically sized only at runtime
# (nhits, maxNumberOfDoublets_) -- become `List[T]`, which is the direct Mojo
# equivalent for an owned, runtime-length, RAII-freed array, and are handed to
# the kernels as `Span`s.
#
# `launch_kernels`/`classify_tuples`/`build_doublets`/`fill_hit_det_indices`/
# `allocate_on_gpu`/`cleanup`/`print_counters` are real methods on this struct
# (matching how C++ calls them: `kernels.launchKernels(...)`, not a free
# function taking the kernels object as an argument) -- their bodies were
# originally going to live in separate Alloc/Impl-mirroring files with the
# low-level kernel_* helpers imported in, but that produced a real circular-
# import type-identity bug (see the comment above `Kernel_checkOverflows`),
# so everything ended up in this one file instead.
struct CAHitNtupletGeneratorKernelsCPU(Movable):
    comptime QualityCuts = QualityCuts
    comptime Params = Params
    comptime Counters = Counters

    comptime HitsView = TrackingRecHit2DHeterogeneous
    comptime HitsOnGPU = TrackingRecHit2DHeterogeneous
    # mojo version of TrackingRecHit2DHeterogeneous is not generic
    comptime HitsOnCPU = TrackingRecHit2DHeterogeneous
    comptime HitToTuple = HitToTuple
    comptime TupleMultiplicity = TupleMultiplicity
    comptime Quality = Quality
    comptime TkSoA = TkSoA
    comptime HitContainer = HitContainer

    # C++ stores `Counters* counters_` pointing at the owning generator's
    # counters. A Mojo struct cannot hold a reference field, and this object is
    # rebuilt per event, so the counters are taken as a `mut` argument by the
    # methods that touch them and keep accumulating on the generator.

    # --- Workspace ---
    # C++ carves both vectors' buffers out of one `cellStorage_` block and
    # hands them to `construct`; SimpleVector owns its storage here, so the
    # block and the two container pointers are gone.
    var device_theCellNeighbors_: CAConstants.CellNeighborsVector
    var device_theCellTracks_: CAConstants.CellTracksVector

    var device_theCells_: List[GPUCACell]
    var device_isOuterHitOfCell_: CAConstants.OuterHitOfCellContainer
    # C++ carves these three out of one 3-slot allocation, punning the third
    # from AtomicPairCounter to uint32. Each is its own typed field here.
    var device_nCells_: UInt32

    var device_hitToTuple_: HitToTuple
    var device_hitToTuple_apc_: AtomicPairCounter

    var device_hitTuple_apc_: AtomicPairCounter

    var device_tupleMultiplicity_: TupleMultiplicity

    # C++ holds `Params const& m_params` (bound to the caller's Params, which
    # outlives the kernels object). Mojo structs can't hold reference-typed
    # fields, so this stores an owned copy instead -- Params is never mutated
    # after construction, so the difference is not observable.
    var m_params: Params

    def __init__(out self, params: Params):
        self.device_theCellNeighbors_ = CAConstants.CellNeighborsVector()
        self.device_theCellTracks_ = CAConstants.CellTracksVector()

        self.device_theCells_ = List[GPUCACell]()
        self.device_isOuterHitOfCell_ = CAConstants.OuterHitOfCellContainer()
        self.device_nCells_ = 0

        self.device_hitToTuple_ = HitToTuple()
        self.device_hitToTuple_apc_ = AtomicPairCounter()

        self.device_hitTuple_apc_ = AtomicPairCounter()

        self.device_tupleMultiplicity_ = TupleMultiplicity()

        self.m_params = params

    def __init__(out self, *, deinit move: Self):
        self.device_theCellNeighbors_ = move.device_theCellNeighbors_^
        self.device_theCellTracks_ = move.device_theCellTracks_^
        self.device_theCells_ = move.device_theCells_^
        self.device_isOuterHitOfCell_ = move.device_isOuterHitOfCell_^
        self.device_nCells_ = move.device_nCells_
        self.device_hitToTuple_ = move.device_hitToTuple_^
        self.device_hitToTuple_apc_ = move.device_hitToTuple_apc_
        self.device_hitTuple_apc_ = move.device_hitTuple_apc_
        self.device_tupleMultiplicity_ = move.device_tupleMultiplicity_^
        self.m_params = move.m_params

    def tuple_multiplicity(
        ref self,
    ) -> ref [self.device_tupleMultiplicity_] TupleMultiplicity:
        return self.device_tupleMultiplicity_

    # C++: CAHitNtupletGeneratorKernelsCPU::allocateOnGPU (CAHitNtupletGeneratorKernelsAlloc.h)
    def allocate_on_gpu(mut self, stream: Stream):
        #//////////////////////////////////////////////////////////
        #// ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
        #//////////////////////////////////////////////////////////

        # C++: Traits::template make_unique<T>(stream), which for CPUTraits
        # (see HeterogeneousSoA.h) is just std::make_unique<T>() -- the
        # stream parameter exists for the GPU trait and is unused on CPU.
        # Ported directly as fresh construction, same as __init__.
        self.device_theCellNeighbors_ = CAConstants.CellNeighborsVector()
        self.device_theCellTracks_ = CAConstants.CellTracksVector()
        self.device_hitToTuple_ = HitToTuple()
        self.device_tupleMultiplicity_ = TupleMultiplicity()

        self.device_hitTuple_apc_ = AtomicPairCounter()
        self.device_hitToTuple_apc_ = AtomicPairCounter()
        self.device_nCells_ = 0

        launchZero(self.device_tupleMultiplicity_)
        launchZero(self.device_hitToTuple_)  # we may wish to keep it in the edm...

    # C++: CAHitNtupletGeneratorKernelsCPU::buildDoublets (CAHitNtupletGeneratorKernels.cc)
    def build_doublets(mut self, hh: Self.HitsOnCPU, stream: Stream) raises:
        var nhits = hh.nHits()

        comptime if is_defined["NTUPLE_DEBUG"]():
            print("building Doublets out of", nhits, "Hits")

        # in principle we can use "nhits" to heuristically dimension the
        # workspace... overkill to use template here (std::make_unique would suffice)
        gpuPixelDoublets.initDoublets(
            self.device_theCellNeighbors_,
            self.device_theCellTracks_,
        )

        # device_theCells_ = Traits:: template make_unique<GPUCACell[]>(cs, m_params.maxNumberOfDoublets_, stream);
        self.device_theCells_ = List[GPUCACell](
            length=Int(self.m_params.maxNumberOfDoublets), fill=GPUCACell()
        )
        if nhits == 0:
            return  # protect against empty events

        # FIXME avoid magic numbers
        var nActualPairs = gpuPixelDoublets.nPairs
        if not self.m_params.includeJumpingForwardDoublets:
            nActualPairs = 15
        if self.m_params.minHitsPerNtuplet > 3:
            nActualPairs = 13

        debug_assert(nActualPairs <= gpuPixelDoublets.nPairs)
        gpuPixelDoublets.getDoubletsFromHisto(
            Span(self.device_theCells_),
            self.device_nCells_,
            self.device_theCellNeighbors_,
            self.device_theCellTracks_,
            hh,
            nActualPairs,
            self.m_params.idealConditions,
            self.m_params.doClusterCut,
            self.m_params.doZ0Cut,
            self.m_params.doPtCut,
            self.m_params.maxNumberOfDoublets,
        )

        gpuPixelDoublets.fillOuterHitOfCell(
            Span(self.device_theCells_),
            self.device_nCells_,
            nhits,
            self.device_isOuterHitOfCell_,
        )

    # C++: CAHitNtupletGeneratorKernelsCPU::launchKernels (CAHitNtupletGeneratorKernels.cc)
    def launch_kernels(
        mut self,
        hh: Self.HitsOnCPU,
        mut tracks_d: TkSoA,
        cudaStream: Stream,
        mut counters: Counters,
    ) raises:
        ref tuples_d = tracks_d.hitIndices
        # field path, not tracks_d.qualityData(): that takes `ref self`, so its
        # Span borrows the whole struct and aliases tuples_d
        var quality_d = Span(tracks_d.m_quality._data)

        # zero tuples
        launchZero(tuples_d)

        var nhits = hh.nHits()
        debug_assert(nhits <= GPUClusteringConstants.maxNumberOfHits)

        #
        # applying conbinatoric cleaning such as fishbone at this stage is too expensive
        #

        kernel_connect(
            self.device_hitTuple_apc_,
            self.device_hitToTuple_apc_,  # needed only to be reset, ready for next kernel
            hh,
            Span(self.device_theCells_),
            self.device_nCells_,
            self.device_theCellNeighbors_,
            self.device_isOuterHitOfCell_,
            self.m_params.hardCurvCut,
            self.m_params.ptmin,
            self.m_params.CAThetaCutBarrel,
            self.m_params.CAThetaCutForward,
            self.m_params.dcaCutInnerTriplet,
            self.m_params.dcaCutOuterTriplet,
        )

        if nhits > 1 and self.m_params.earlyFishbone:
            gpuPixelDoublets.fishbone(
                hh,
                Span(self.device_theCells_),
                self.device_nCells_,
                self.device_theCellTracks_,
                self.device_isOuterHitOfCell_,
                nhits,
                False,
            )

        kernel_find_ntuplets(
            hh,
            Span(self.device_theCells_),
            self.device_nCells_,
            self.device_theCellNeighbors_,
            self.device_theCellTracks_,
            tuples_d,
            self.device_hitTuple_apc_,
            quality_d,
            self.m_params.minHitsPerNtuplet,
        )
        if self.m_params.doStats:
            kernel_mark_used(
                hh,
                Span(self.device_theCells_),
                self.device_nCells_,
                self.device_theCellTracks_,
            )

        finalizeBulk(self.device_hitTuple_apc_, tuples_d)

        # remove duplicates (tracks that share a doublet)
        kernel_earlyDuplicateRemover(
            Span(self.device_theCells_),
            self.device_nCells_,
            self.device_theCellTracks_,
            tuples_d,
            quality_d,
        )

        kernel_countMultiplicity(
            tuples_d, quality_d, self.device_tupleMultiplicity_
        )
        launchFinalize(self.device_tupleMultiplicity_)
        kernel_fillMultiplicity(
            tuples_d, quality_d, self.device_tupleMultiplicity_
        )

        if nhits > 1 and self.m_params.lateFishbone:
            gpuPixelDoublets.fishbone(
                hh,
                Span(self.device_theCells_),
                self.device_nCells_,
                self.device_theCellTracks_,
                self.device_isOuterHitOfCell_,
                nhits,
                True,
            )

        if self.m_params.doStats:
            Kernel_checkOverflows(
                tuples_d,
                self.device_tupleMultiplicity_,
                self.device_hitTuple_apc_,
                Span(self.device_theCells_),
                self.device_nCells_,
                self.device_theCellNeighbors_,
                self.device_theCellTracks_,
                self.device_isOuterHitOfCell_,
                nhits,
                self.m_params.maxNumberOfDoublets,
                counters,
            )

    # C++: CAHitNtupletGeneratorKernelsCPU::classifyTuples (CAHitNtupletGeneratorKernels.cc)
    def classify_tuples(
        mut self,
        hh: Self.HitsOnCPU,
        mut tracks_d: TkSoA,
        cudaStream: Stream,
        mut counters: Counters,
    ) raises:
        ref tuples_d = tracks_d.hitIndices
        # field path, not tracks_d.qualityData(): that takes `ref self`, so its
        # Span borrows the whole struct and aliases tuples_d
        var quality_d = Span(tracks_d.m_quality._data)

        # classify tracks based on kinematics
        kernel_classifyTracks(tracks_d, self.m_params.cuts)

        if self.m_params.lateFishbone:
            # apply fishbone cleaning to good tracks
            kernel_fishboneCleaner(
                Span(self.device_theCells_),
                self.device_nCells_,
                self.device_theCellTracks_,
                quality_d,
            )

        # remove duplicates (tracks that share a doublet)
        kernel_fastDuplicateRemover(
            Span(self.device_theCells_),
            self.device_nCells_,
            self.device_theCellTracks_,
            tracks_d,
        )

        # fill hit->track "map"
        kernel_countHitInTracks(
            tuples_d, quality_d, self.device_hitToTuple_
        )
        launchFinalize(self.device_hitToTuple_)
        kernel_fillHitInTracks(
            tuples_d, quality_d, self.device_hitToTuple_
        )

        # remove duplicates (tracks that share a hit)
        kernel_tripletCleaner(
            hh,
            tracks_d,
            self.device_hitToTuple_,
        )

        if self.m_params.doStats:
            # counters (add flag???)
            kernel_doStatsForHitInTracks(
                self.device_hitToTuple_, counters
            )
            kernel_doStatsForTracks(tuples_d, quality_d, counters)

        comptime if is_defined["DUMP_GPU_TK_TUPLES"]():
            # C++ keeps a static atomic `iev` counter across calls to number
            # the printed events; Mojo has no mutable module-level globals,
            # and this dump path is off by default (disabled-debug-only), so
            # the printed event index is not persisted across calls here.
            kernel_print_found_ntuplets(
                hh,
                tracks_d,
                self.device_hitToTuple_,
                100,
                0,
            )

    # C++: CAHitNtupletGeneratorKernelsCPU::fillHitDetIndices (CAHitNtupletGeneratorKernels.cc)
    def fill_hit_det_indices(
        self,
        hh: TrackingRecHit2DHeterogeneous,
        mut tracks_d: TkSoA,
        cudaStream: Stream,
    ) raises:
        kernel_fillHitDetIndices(
            tracks_d.hitIndices,
            hh,
            tracks_d.detIndices,
        )

    # C++ declares this method but never defines or calls it anywhere in the
    # serial backend -- the owned fields free themselves when
    # `self` is destroyed, so there's nothing left to free manually here.
    def cleanup(self, cudaStream: Stream):
        pass

    @staticmethod
    def print_counters(counters: Counters):
        kernel_printCounters(counters)

    @staticmethod
    def dtype() -> String:
        return "CAHitNtupletGeneratorKernelsCPU"
