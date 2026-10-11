import std.math as math
import MojoSerial.plugin_PixelTriplets.CAConstants as CAConstants
import MojoSerial.plugin_PixelTriplets.BrokenLine as BrokenLine
from MojoSerial.plugin_PixelTriplets.HelixFitOnGPU import Rfit
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import PixelTrack as pixelTrack
from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from std.atomic import Atomic
from MojoSerial.MojoBridge.Matrix import to_layout_tensor


comptime HitsOnGPU = TrackingRecHit2DHeterogeneous
comptime Tuples = pixelTrack.HitContainer
comptime OutputSoA = pixelTrack.TrackSoA

comptime BROKENLINE_DEBUG = False

comptime BL_DUMP_HITS = False


# The fast fit and the fit of each n-tuplet, fused as in the serial back-end: the hits of the n-tuplet are kept in
# local matrices, instead of the strided buffers used to coalesce the memory accesses on GPUs.
# C++ reads `hhp->cpeParams()`; that field went with the view (§11).
def kernelBLFastFitAndFit[N: Int](
    tupleMultiplicity: CAConstants.TupleMultiplicity,
    hhp: HitsOnGPU,
    cpeParams: PixelCPEFast,
    B: Float64,
    mut results: OutputSoA,
    nHits: UInt32,
    maxTuples: UInt32,
):
    var hitsInFit: UInt32 = UInt32(N)
    debug_assert(hitsInFit <= nHits)

    comptime if BROKENLINE_DEBUG:
        var nbins_val = results.hitIndices.nbins()
        print(nbins_val, "total Ntuple")
        var tsize_val = tupleMultiplicity.size(nHits)
        print(
            tsize_val, "Ntuple of size", nHits, "for", hitsInFit, "hits to fit"
        )

    var nt = min(Int(tupleMultiplicity.size(nHits)), Int(maxTuples))
    for tuple_idx in range(nt):
        var tkid = UInt32(tupleMultiplicity.begin(nHits)[tuple_idx])
        debug_assert(tkid < results.hitIndices.nbins())
        debug_assert(results.hitIndices.size(tkid) == nHits)

        var hits = Rfit.Matrix3xNd[N](uninitialized=True)
        var fast_fit = Rfit.Vector4d(uninitialized=True)
        var hits_ge = Rfit.Matrix6xNf[N](uninitialized=True)

        var hitId = results.hitIndices.begin(tkid)
        for idx in range(N):
            var hit = Int(hitId[idx])
            var ge = InlineArray[Float32, 6](uninitialized=True)
            cpeParams.detParams(Int(hhp.detectorIndex(hit))).frame.toGlobal(
                hhp.xerrLocal(hit),
                0,
                hhp.yerrLocal(hit),
                Span(ge),
            )
            hits[0, idx] = hhp.xGlobal(hit).cast[DType.float64]()
            hits[1, idx] = hhp.yGlobal(hit).cast[DType.float64]()
            hits[2, idx] = hhp.zGlobal(hit).cast[DType.float64]()
            hits_ge[0, idx] = ge[0]
            hits_ge[1, idx] = ge[1]
            hits_ge[2, idx] = ge[2]
            hits_ge[3, idx] = ge[3]
            hits_ge[4, idx] = ge[4]
            hits_ge[5, idx] = ge[5]

        BrokenLine.BL_Fast_fit(hits, fast_fit)

        debug_assert(fast_fit[0] == fast_fit[0])
        debug_assert(fast_fit[1] == fast_fit[1])
        debug_assert(fast_fit[2] == fast_fit[2])
        debug_assert(fast_fit[3] == fast_fit[3])

        var data = BrokenLine.PreparedBrokenLineData[N]()
        var circle = BrokenLine.karimaki_circle_fit()
        var line = Rfit.line_fit()

        BrokenLine.prepareBrokenLineData(hits, fast_fit, B, data)
        BrokenLine.BL_Line_fit(hits_ge, fast_fit, B, data, line)
        BrokenLine.BL_Circle_fit(hits, hits_ge, fast_fit, B, data, circle)

        var track_idx = Int(tkid)
        var cp_buf = InlineArray[Scalar[DType.float64], 3](uninitialized=True)
        var ccov_buf = InlineArray[Scalar[DType.float64], 9](uninitialized=True)
        var lp_buf = InlineArray[Scalar[DType.float64], 2](uninitialized=True)
        var lcov_buf = InlineArray[Scalar[DType.float64], 4](uninitialized=True)
        results.stateAtBS.copyFromCircle(
            to_layout_tensor(circle.par, cp_buf),
            to_layout_tensor(circle.cov, ccov_buf),
            to_layout_tensor(line.par, lp_buf),
            to_layout_tensor(line.cov, lcov_buf),
            Float32(1.0 / B),
            Int32(track_idx),
        )
        results.pt[track_idx] = Float32(B) / Float32(abs(circle.par[2]))
        results.eta[track_idx] = math.asinh(Float32(line.par[0]))
        var chi2 = Float64(circle.chi2) + line.chi2
        results.chi2[track_idx] = Float32(
            chi2 / Float64(2 * N - 5)
        )

        comptime if BROKENLINE_DEBUG:
            if not (circle.chi2 >= 0) or not (line.chi2 >= 0):
                print("kernelBLFit failed!", circle.chi2, "/", line.chi2)
            print(
                "kernelBLFit size", N, "for", nHits,
                "hits circle.par(0,1,2):", tkid,
                circle.par[0], ",", circle.par[1], ",", circle.par[2],
            )
