from std.sys import is_defined

from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.CUDACore.HistoContainer import HistoContainer, forEachInBins
from MojoSerial.MojoBridge.DTypes import Float
from MojoSerial.plugin_PixelVertexFinding.gpuVertexFinder import ZVertices, WorkSpace

comptime verbose: Bool = False  # in principle the compiler should optmize out if false


# this algo does not really scale as it works in a single block...
# enough for <10K tracks we have
@always_inline
def clusterTracksIterative(
    mut data: ZVertices,
    mut ws: WorkSpace,
    minT: Int32,  # min number of neighbours to be "core"
    eps: Float,  # max absolute distance to cluster
    errmax: Float,  # max error to be "seed"
    chi2max: Float,  # max normalized distance to cluster
):
    comptime if verbose:
        print("params", minT, eps, errmax, chi2max)

    var er2mx = errmax * errmax

    var nt = ws.ntrks
    ref zt = ws.zt
    ref ezt2 = ws.ezt2

    ref nvFinal = data.nvFinal
    ref nvIntermediate = ws.nvIntermediate

    ref izt = ws.izt
    ref nn = data.ndof
    ref iv = ws.iv

    comptime Hist = HistoContainer[DType.uint8, 256, 16000, 8, DType.uint16, 1]
    var hist: Hist = Hist()

    for j in range(Hist.totbins()):
        hist.off[j] = 0

    comptime if verbose:
        print(
            "booked hist with",
            Hist.nbins(),
            "bins, size",
            Hist.capacity(),
            "for",
            nt,
            "tracks",
        )

    debug_assert(nt <= Hist.capacity())

    # fill hist  (bin shall be wider than "eps")
    for i in range(nt):
        debug_assert(i < ZVertices.MAXTRACKS)
        var iz: Int32 = Int32(zt[i] * 10.0)  # valid if eps<=0.1
        iz = min(max(iz, Int32(Int8.MIN)), Int32(Int8.MAX))
        izt[i] = UInt8(iz - Int32(Int8.MIN))
        debug_assert(iz - Int32(Int8.MIN) >= 0)
        debug_assert(iz - Int32(Int8.MIN) < 256)
        hist.count(izt[i])
        iv[i] = Int32(i)
        nn[i] = 0

    hist.finalize()

    debug_assert(hist.size() == nt)
    for i in range(nt):
        hist.fill(izt[i], UInt16(i))

    # count neighbours
    for i in range(nt):
        if ezt2[i] > er2mx:
            continue

        @parameter
        def countNeighbor(j: UInt16):
            if i == UInt32(j):
                return
            var dist = abs(zt[i] - zt[Int(j)])
            if dist > eps:
                return
            if dist * dist > chi2max * (ezt2[i] + ezt2[Int(j)]):
                return
            nn[i] += 1

        forEachInBins[func=countNeighbor](hist, izt[i], 1)

    var nloops: Int32 = 0

    # cluster seeds only
    var more: Bool = True
    while more:
        if nloops % 2 == 1:
            for i in range(nt):
                var m = iv[i]
                while m != iv[Int(m)]:
                    m = iv[Int(m)]
                iv[i] = m
        else:
            more = False
            for k in range(hist.size()):
                var p = hist.begin() + Int(k)
                var i = Int(p[])
                var be = min(Int(Hist.bin(izt[i])) + 1, Int(Hist.nbins()) - 1)
                if nn[i] < minT:
                    continue  # DBSCAN core rule

                @parameter
                def propagateMin(j: UInt16):
                    debug_assert(i != Int(j))
                    if nn[Int(j)] < minT:
                        return  # DBSCAN core rule
                    var dist = abs(zt[i] - zt[Int(j)])
                    if dist > eps:
                        return
                    if dist * dist > chi2max * (ezt2[i] + ezt2[Int(j)]):
                        return
                    var old = CUDACompat.atomicMin(iv[Int(j)], iv[i])
                    if old != iv[i]:
                        # end the loop only if no changes were applied
                        more = True
                    _ = CUDACompat.atomicMin(iv[i], old)

                p += 1
                while p < hist.end(UInt32(be)):
                    propagateMin(p[])
                    p += 1
        nloops += 1

    # collect edges (assign to closest cluster of closest point??? here to closest point)
    for i in range(nt):
        #    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
        if nn[i] >= minT:
            continue  # DBSCAN edge rule
        var mdist: Float = eps

        @parameter
        def closestCore(j: UInt16):
            if nn[Int(j)] < minT:
                return  # DBSCAN core rule
            var dist = abs(zt[i] - zt[Int(j)])
            if dist > mdist:
                return
            if dist * dist > chi2max * (ezt2[i] + ezt2[Int(j)]):
                return  # needed?
            mdist = dist
            iv[i] = iv[Int(j)]  # assign to cluster (better be unique??)

        forEachInBins[func=closestCore](hist, izt[i], 1)

    var foundClusters: UInt32 = 0

    # find the number of different clusters, identified by a tracks with clus[i] == i;
    # mark these tracks with a negative id.
    for i in range(nt):
        if iv[i] == Int32(i):
            if nn[i] >= minT:
                var old = CUDACompat.atomicInc(
                    foundClusters, UInt32(0xFFFFFFFF)
                )
                iv[i] = -(Int32(old) + 1)
            else:  # noise
                iv[i] = -9998

    debug_assert(foundClusters < ZVertices.MAXVTX)

    # propagate the negative id to all the tracks in the cluster.
    for i in range(nt):
        if iv[i] >= 0:
            # mark each track in a cluster with the same id as the first one
            iv[i] = iv[Int(iv[i])]

    # adjust the cluster id to be a positive value starting from 0
    for i in range(nt):
        iv[i] = -iv[i] - 1

    nvIntermediate = foundClusters
    nvFinal = foundClusters

    comptime if verbose:
        print("found", foundClusters, "proto vertices")
