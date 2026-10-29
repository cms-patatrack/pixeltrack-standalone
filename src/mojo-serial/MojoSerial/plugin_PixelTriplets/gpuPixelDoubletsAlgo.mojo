import MojoSerial.plugin_PixelTriplets.CAConstants as CAConstants
from MojoSerial.plugin_PixelTriplets.GPUCACell import GPUCACell
from std.sys import is_defined
from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
    Hist,
)
from MojoSerial.DataFormats.ApproxAtan2 import ApproxAtan2

comptime CellNeighbors = CAConstants.CellNeighbors
comptime CellTracks = CAConstants.CellTracks
comptime CellNeighborsVector = CAConstants.CellNeighborsVector
comptime CellTracksVector = CAConstants.CellTracksVector


def doubletsFromHisto(
    layerPairs: Span[UInt8, _],
    nPairs: UInt32,
    cells: Span[mut=True, GPUCACell, _],
    mut nCells: UInt32,
    mut cellNeighbors: CellNeighborsVector,
    mut cellTracks: CellTracksVector,
    hh: TrackingRecHit2DHeterogeneous,
    isOuterHitOfCell: Span[mut=True, GPUCACell.OuterHitOfCell, _],
    phicuts: Span[Int16, _],
    minz: Span[Float32, _],
    maxz: Span[Float32, _],
    maxr: Span[Float32, _],
    ideal_cond: Bool,
    doClusterCut: Bool,
    doZ0Cut: Bool,
    doPtCut: Bool,
    maxNumOfDoublets: UInt32,
):
    # ysize cuts (z in the barrel) times 8
    # these are used if doClusterCut is true
    comptime minYsizeB1: Int = 36
    comptime minYsizeB2: Int = 28
    comptime maxDYsize12: Int = 28
    comptime maxDYsize: Int = 20
    comptime maxDYPred: Int = 20
    comptime dzdrFact: Float32 = 8.0 * 0.0285 / 0.015  # from dz/dr to "DY"

    var isOuterLadder: Bool = ideal_cond

    ref hist = hh.phiBinner()
    # hitsLayerStart() already returns a Span; C++ asserts it is non-null, which
    # a Span cannot be.
    var offsets = hh.hitsLayerStart()

    @parameter
    def layerSize(li: UInt8) -> UInt32:
        var idx = Int(li)
        return offsets[idx + 1] - offsets[idx]

    # nPairsMax to be optimized later (originally was 64).
    # If it should be much bigger, consider using a block-wide parallel prefix scan,
    # e.g. see https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    comptime nPairsMax: Int = Int(CAConstants.maxNumberOfLayerPairs())
    debug_assert(nPairs <= UInt32(nPairsMax))
    var innerLayerCumulativeSize = InlineArray[UInt32, nPairsMax](uninitialized=True)
    var ntot: UInt32 = 0

    innerLayerCumulativeSize[0] = layerSize(layerPairs[0])
    for i in range(1, Int(nPairs)):
        innerLayerCumulativeSize[i] = (
            innerLayerCumulativeSize[i - 1]
            + layerSize(layerPairs[2 * i])
        )
    ntot = innerLayerCumulativeSize[Int(nPairs) - 1]

    # x runs faster
    var idy = 0
    var first: UInt32 = 0
    var stride = 1

    var pairLayerId: UInt32 = 0  # cannot go backward
    var j: UInt32 = UInt32(idy)
    while j < ntot:
        while j >= innerLayerCumulativeSize[Int(pairLayerId)]:
            pairLayerId += 1

        debug_assert(pairLayerId < nPairs)
        debug_assert(j < innerLayerCumulativeSize[Int(pairLayerId)])
        if pairLayerId > 0:
            debug_assert(j >= innerLayerCumulativeSize[Int((pairLayerId - 1))])
        else:
            debug_assert(pairLayerId == 0)

        var inner :UInt8 = layerPairs[Int((2 * pairLayerId))]
        var outer :UInt8= layerPairs[Int((2 * pairLayerId + 1))]
        debug_assert(outer > inner)
        var inner_idx = Int(inner)
        var outer_idx = Int(outer)

        var hoff = Hist.histOff(UInt32(outer))

        var i = j
        if pairLayerId != 0:
            i -= innerLayerCumulativeSize[Int((pairLayerId - 1))]
        i += offsets[inner_idx]

        # printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

        debug_assert(i >= offsets[inner_idx])
        debug_assert(i < offsets[inner_idx + 1])

        # found hit corresponding to our cuda thread, now do the job
        var mi = hh.detectorIndex(Int(i))
        if mi > 2000:
            j += 1
            continue  # invalid

        # maybe clever, not effective when zoCut is on
        # auto bpos = (mi%8)/4  # if barrel is 1 for z>0
        # auto fpos = (outer>3) & (outer<7)
        # if ((inner<3) & (outer>3)) and bpos!=fpos: continue

        var mez = hh.zGlobal(Int(i))

        if mez < minz[Int(pairLayerId)] or mez > maxz[Int(pairLayerId)]:
            j += 1
            continue

        var mes: Int16 = -1
        if doClusterCut:
            # if ideal treat inner ladder as outer
            if inner == 0:
                debug_assert(mi < 96)
            if ideal_cond:
                isOuterLadder = True
            else:
                isOuterLadder = (UInt32((mi / 8)) % 2) == 0

            # in any case we always test mes>0 ...
            if inner > 0 or isOuterLadder:
                mes = hh.clusterSizeY(Int(i))
            else:
                mes = -1

            if inner == 0 and outer > 3:  # B1 and F1
                var mes_i = Int(mes)
                if mes_i > 0 and mes_i < minYsizeB1:
                    j += 1
                    continue  # only long cluster  (5*8)
            if inner == 1 and outer > 3:  # B2 and F1
                var mes_i = Int(mes)
                if mes_i > 0 and mes_i < minYsizeB2:
                    j += 1
                    continue

        var mep = hh.iphi(Int(i))
        var mer = hh.rGlobal(Int(i))

        # all cuts: true if fails
        comptime z0cut: Float32 = 12.0  # cm
        comptime hardPtCut: Float32 = 0.5  # GeV
        comptime minRadius: Float32 = hardPtCut * 87.78  # cm
        comptime minRadius2T4: Float32 = 4.0 * minRadius * minRadius

        # `hh` is passed in rather than captured: nested closures capture by
        # copy, and the hit SoA is not Copyable.
        @parameter
        def ptcut(
            hh: TrackingRecHit2DHeterogeneous, j: Int, idphi: Int
        ) -> Bool:
            var r2t4 = minRadius2T4
            var ri = mer
            var ro = hh.rGlobal(j)
            var dphi = ApproxAtan2.short2phi(Int16(idphi))
            return dphi * dphi * (r2t4 - ri * ro) > (ro - ri) * (ro - ri)

        @parameter
        def z0cutoff(hh: TrackingRecHit2DHeterogeneous, j: Int) -> Bool:
            var zo = hh.zGlobal(j)
            var ro = hh.rGlobal(j)
            var dr = ro - mer
            return dr > maxr[Int(pairLayerId)] or dr < 0.0 or abs(mez * ro - mer * zo) > z0cut * dr

        @parameter
        def zsizeCut(hh: TrackingRecHit2DHeterogeneous, j: Int) -> Bool:
            var onlyBarrel = outer < 4
            var so = hh.clusterSizeY(j)
            var dy = maxDYsize12 if inner == 0 else maxDYsize
            # in the barrel cut on difference in size
            # in the endcap on the prediction on the first layer (actually in the barrel only: happen to be safe for endcap as well)
            # FIXME move pred cut to z0cutoff to optmize loading of and computaiton ...
            var zo = hh.zGlobal(j)
            var ro = hh.rGlobal(j)
            if onlyBarrel:
                var mes_i = Int(mes)
                var so_i = Int(so)
                return mes_i > 0 and so_i > 0 and abs(so_i - mes_i) > dy
            var mes_i = Int(mes)
            return inner < 4 and mes_i > 0 and abs(mes_i -  Int((abs((mez - zo) / (mer - ro)) * dzdrFact + 0.5))) > maxDYPred


        var iphicut = phicuts[Int(pairLayerId)]
        var kl = UInt32(Hist.bin(Int16((mep - iphicut))))
        var kh = UInt32(Hist.bin(Int16((mep + iphicut))))
        def incr(mut k: UInt32) -> UInt32:
            k = (k + 1) % Hist.nbins()
            return k

        # Declared unconditionally -- each @parameter if is its own isolated
        # scope in Mojo (confirmed even adjacent blocks can't share a
        # variable), so these can't be declared conditionally and still be
        # reachable from the increments below and the print after the loop.
        # Only ever read or written under the GPU_DEBUG guard.
        var tot: Int = 0
        var nmin: Int = 0
        var tooMany: Int = 0

        var kk = kl
        var khh = kh
        _ = incr(khh)
        while kk != khh:
            comptime if is_defined["GPU_DEBUG"]():
                if kk != kl and kk != kh:
                    nmin += Int(hist.size(kk + hoff))
            var p = hist.begin(kk + hoff)
            var e = hist.end(kk + hoff)
            p += Int(first)
            while p < e:
                var oi = p[]
                var oi_i = Int(oi)
                var oi_u = UInt32(oi)
                debug_assert(oi_u >= offsets[outer_idx])
                debug_assert(oi_u < offsets[outer_idx + 1])
                var mo = hh.detectorIndex(oi_i)
                if mo > 2000:
                    p += stride
                    continue  # invalid

                if doZ0Cut and z0cutoff(hh, oi_i):
                        p += stride
                        continue

                var mop = hh.iphi(oi_i)

                var idphi = min(abs(Int((mop - mep))), abs(Int((mep - mop))))
                if idphi > Int(iphicut):
                    p += stride
                    continue

                if doClusterCut and zsizeCut(hh, oi_i):
                        p += stride
                        continue

                if doPtCut and ptcut(hh, oi_i, idphi):
                        p += stride
                        continue

                var ind = CUDACompat.atomicAdd(nCells, UInt32(1))
                if ind >= maxNumOfDoublets:
                    _ = CUDACompat.atomicSub(nCells, UInt32(1))
                    break
                cells[Int(ind)].init(
                    cellNeighbors,
                    cellTracks,
                    hh,
                    Int32(pairLayerId),
                    Int32(ind),
                    i.cast[DType.uint16](),
                    oi.cast[DType.uint16](),
                )
                isOuterHitOfCell[oi_i].push_back(ind)
                comptime if is_defined["GPU_DEBUG"]():
                    if isOuterHitOfCell[oi_i].full():
                        tooMany += 1
                    tot += 1
                p += stride

            _ = incr(kk)

        comptime if is_defined["GPU_DEBUG"]():
            if tooMany > 0:
                print(
                    "OuterHitOfCell full for ",
                    i,
                    " in layer ",
                    inner,
                    "/",
                    outer,
                    ", ",
                    nmin,
                    ",",
                    tot,
                    " ",
                    tooMany,
                )

        j += 1
