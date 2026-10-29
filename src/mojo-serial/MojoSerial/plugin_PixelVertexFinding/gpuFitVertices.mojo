from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.MojoBridge.DTypes import Float
from MojoSerial.plugin_PixelVertexFinding.gpuVertexFinder import ZVertices, WorkSpace

comptime verbose: Bool = False  # in principle the compiler should optmize out if false


@always_inline
def fitVertices(
    mut data: ZVertices,
    mut ws: WorkSpace,
    chi2Max: Float,  # for outlier rejection
):
    var nt = ws.ntrks
    ref zt = ws.zt
    ref ezt2 = ws.ezt2
    ref zv = data.zv
    ref wv = data.wv
    ref chi2 = data.chi2
    ref nvFinal = data.nvFinal
    var nvIntermediate: UInt32 = ws.nvIntermediate

    ref nn = data.ndof
    ref iv = ws.iv

    debug_assert(nvFinal <= nvIntermediate)
    nvFinal = nvIntermediate
    var foundClusters = nvFinal

    # zero
    for i in range(foundClusters):
        zv[i] = 0
        wv[i] = 0
        chi2[i] = 0

    # only for test
    var noise: Int32 = 0

    comptime if verbose:
        noise = 0

    # compute cluster location
    for i in range(nt):
        if iv[i] > 9990:

            comptime if verbose:
                _ = CUDACompat.atomicAdd(noise, Int32(1))
            continue
        debug_assert(iv[i] >= 0)
        debug_assert(iv[i] < Int32(foundClusters))
        var w = 1.0 / ezt2[i]
        _ = CUDACompat.atomicAdd(zv[Int(iv[i])], zt[i] * w)
        _ = CUDACompat.atomicAdd(wv[Int(iv[i])], w)

    # reuse nn
    for i in range(foundClusters):
        debug_assert(wv[i] > 0.0)
        zv[i] /= wv[i]
        nn[i] = -1  # ndof

    # compute chi2
    for i in range(nt):
        if iv[i] > 9990:
            continue

        var c2 = zv[Int(iv[i])] - zt[i]
        c2 *= c2 / ezt2[i]
        if c2 > chi2Max:
            iv[i] = 9999
            continue
        _ = CUDACompat.atomicAdd(chi2[Int(iv[i])], c2)
        _ = CUDACompat.atomicAdd(nn[Int(iv[i])], Int32(1))

    for i in range(foundClusters):
        if nn[i] > 0:
            wv[i] *= Float(nn[i]) / chi2[i]

    comptime if verbose:
        print("found", foundClusters, "proto clusters ")

    comptime if verbose:
        print("and", noise, "noise")


@always_inline
def fitVerticesKernel(
    mut data: ZVertices,
    mut ws: WorkSpace,
    chi2Max: Float,  # for outlier rejection
):
    fitVertices(data, ws, chi2Max)
