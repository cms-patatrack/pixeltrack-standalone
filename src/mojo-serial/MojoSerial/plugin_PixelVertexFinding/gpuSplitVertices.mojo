from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.MojoBridge.DTypes import Float
from MojoSerial.plugin_PixelVertexFinding.gpuVertexFinder import ZVertices, WorkSpace

comptime verbose: Bool = False  # in principle the compiler should optmize out if false


@always_inline
def splitVertices(mut data: ZVertices, mut ws: WorkSpace, maxChi2: Float):
    var nt = ws.ntrks
    ref zt = ws.zt
    ref ezt2 = ws.ezt2
    ref zv = data.zv
    ref wv = data.wv
    ref chi2 = data.chi2
    ref nvFinal = data.nvFinal

    ref nn = data.ndof
    ref iv = ws.iv

    # one vertex per block
    for kv in range(nvFinal):
        if nn[kv] < 4:
            continue
        if chi2[kv] < maxChi2 * Float(nn[kv]):
            continue

        comptime MAXTK: Int = 512
        debug_assert(nn[kv] < Int32(MAXTK))
        if nn[kv] >= Int32(MAXTK):
            continue  # too bad FIXME
        var it = InlineArray[UInt32, MAXTK](uninitialized=True)  # track index
        var zz = InlineArray[Float, MAXTK](uninitialized=True)  # z pos
        var newV = InlineArray[UInt8, MAXTK](uninitialized=True)  # 0 or 1
        var ww = InlineArray[Float, MAXTK](uninitialized=True)  # z weight

        var nq: UInt32  # number of track for this vertex
        nq = 0

        # copy to local
        for k in range(nt):
            if iv[k] == Int32(kv):
                var old = CUDACompat.atomicInc(nq, UInt32(MAXTK))
                zz[Int(old)] = zt[k] - zv[kv]
                newV[Int(old)] = 0 if zz[Int(old)] < 0 else 1
                ww[Int(old)] = 1.0 / ezt2[k]
                it[Int(old)] = k

        var znew: InlineArray[Float, 2] = [0.0, 0.0]  # the new vertices
        var wnew: InlineArray[Float, 2] = [0.0, 0.0]  # the new vertices

        debug_assert(Int32(nq) == nn[kv] + 1)

        var maxiter: Int32 = 20
        # kt-min....
        var more: Bool = True
        while more:
            more = False

            znew[0] = 0
            znew[1] = 0
            wnew[0] = 0
            wnew[1] = 0

            for k in range(nq):
                var i = newV[Int(k)]
                _ = CUDACompat.atomicAdd(
                    znew[Int(i)], zz[Int(k)] * ww[Int(k)]
                )
                _ = CUDACompat.atomicAdd(wnew[Int(i)], ww[Int(k)])

            znew[0] /= wnew[0]
            znew[1] /= wnew[1]

            for k in range(nq):
                var d0 = abs(zz[Int(k)] - znew[0])
                var d1 = abs(zz[Int(k)] - znew[1])
                var newer: UInt8 = 0 if d0 < d1 else 1
                more |= newer != newV[Int(k)]
                newV[Int(k)] = newer
            maxiter -= 1
            if maxiter <= 0:
                more = False

        # avoid empty vertices
        if wnew[0] == 0 or wnew[1] == 0:
            continue

        # quality cut
        var dist2 = (znew[0] - znew[1]) * (znew[0] - znew[1])

        var chi2Dist = dist2 / (1.0 / wnew[0] + 1.0 / wnew[1])

        comptime if verbose:
            print("inter", 20 - maxiter, chi2Dist, dist2 * wv[kv])

        if chi2Dist < 4:
            continue

        # get a new global vertex
        var igv: UInt32
        igv = CUDACompat.atomicAdd(ws.nvIntermediate, UInt32(1))

        for k in range(nq):
            if newV[Int(k)] == 1:
                iv[Int(it[Int(k)])] = Int32(igv)

    # loop on vertices


@always_inline
def splitVerticesKernel(mut data: ZVertices, mut ws: WorkSpace, maxChi2: Float):
    splitVertices(data, ws, maxChi2)
