from std.sys import is_defined

from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.CUDACore.HistoContainer import HistoContainer, forEachInBins
from MojoSerial.MojoBridge.DTypes import Float
from MojoSerial.plugin_PixelVertexFinding.gpuVertexFinder import ZVertices, WorkSpace

comptime verbose: Bool = False  # in principle the compiler should optmize out if false


@always_inline
def clusterTracksByDensity(
    mut data: ZVertices,
    mut ws: WorkSpace,
    minT: Int32,  # min number of neighbours to be "seed"
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

    # find closest above me .... (we ignore the possibility of two j at same distance from i)
    for i in range(nt):
        var mdist: Float = eps

        @parameter
        def closestAbove(j: UInt16):
            if nn[Int(j)] < nn[i]:
                return
            if nn[Int(j)] == nn[i] and zt[Int(j)] >= zt[i]:
                return  # if equal use natural order...
            var dist = abs(zt[i] - zt[Int(j)])
            if dist > mdist:
                return
            if dist * dist > chi2max * (ezt2[i] + ezt2[Int(j)]):
                return  # (break natural order???)
            mdist = dist
            iv[i] = Int32(j)  # assign to cluster (better be unique??)

        forEachInBins[func=closestAbove](hist, izt[i], 1)

    comptime if is_defined["GPU_DEBUG"]():
        #  mini verification
        for i in range(nt):
            if iv[i] != Int32(i):
                debug_assert(iv[Int(iv[i])] != Int32(i))

    # consolidate graph (percolate index of seed)
    for i in range(nt):
        var m = iv[i]
        while m != iv[Int(m)]:
            m = iv[Int(m)]
        iv[i] = m

    comptime if is_defined["GPU_DEBUG"]():
        #  mini verification
        for i in range(nt):
            if iv[i] != Int32(i):
                debug_assert(iv[Int(iv[i])] != Int32(i))

    comptime if is_defined["GPU_DEBUG"]():
        # and verify that we did not spit any cluster...
        for i in range(nt):
            var minJ = i
            var mdist = eps

            @parameter
            def debugClosest(j: UInt16):
                if nn[Int(j)] < nn[i]:
                    return
                if nn[Int(j)] == nn[i] and zt[Int(j)] >= zt[i]:
                    return  # if equal use natural order...
                var dist = abs(zt[i] - zt[Int(j)])
                if dist > mdist:
                    return
                if dist * dist > chi2max * (ezt2[i] + ezt2[Int(j)]):
                    return
                mdist = dist
                minJ = UInt32(j)

            forEachInBins[func=debugClosest](hist, izt[i], 1)
            # should belong to the same cluster...
            debug_assert(iv[i] == iv[minJ])
            debug_assert(nn[i] <= nn[Int(iv[i])])

    var foundClusters: UInt32 = 0

    # find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
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


@always_inline
def clusterTracksByDensityKernel(
    mut data: ZVertices,
    mut ws: WorkSpace,
    minT: Int32,  # min number of neighbours to be "seed"
    eps: Float,  # max absolute distance to cluster
    errmax: Float,  # max error to be "seed"
    chi2max: Float,  # max normalized distance to cluster
):
    clusterTracksByDensity(data, ws, minT, eps, errmax, chi2max)
