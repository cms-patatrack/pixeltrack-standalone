import std.math as math

import MojoSerial.plugin_PixelTriplets.CAConstants as CAConstants
import MojoSerial.plugin_PixelTriplets.RiemannFit as RiemannFit
from MojoSerial.plugin_PixelTriplets.FitUtils import Rfit as FitRfit
from MojoSerial.plugin_PixelTriplets.HelixFitOnGPU import Rfit
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrack as pixelTrack,
)
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from MojoSerial.MojoBridge.Matrix import to_layout_tensor


comptime RIEMANN_DEBUG = False


comptime HitsOnGPU = TrackingRecHit2DHeterogeneous
comptime Tuples = pixelTrack.HitContainer
comptime OutputSoA = pixelTrack.TrackSoA
comptime CircleFit = FitRfit.circle_fit
comptime LineFit = FitRfit.line_fit

# C++ reads `hhp->cpeParams()`; that field went with the view (§11).
def kernelFastFit[N: Int](
    foundNtuplets: Tuples,
    tupleMultiplicity: CAConstants.TupleMultiplicity,
    nHits: UInt32,
    hhp: HitsOnGPU,
    cpeParams: PixelCPEFast,
    phits: Span[mut=True, Float64, _],
    phits_ge: Span[mut=True, Float32, _],
    pfast_fit: Span[mut=True, Float64, _],
    offset: UInt32,
):
    comptime hitsInFit: UInt32 = UInt32(N)

    debug_assert(hitsInFit <= nHits)
    # C++ asserts pfast_fit/foundNtuplets/tupleMultiplicity are non-null; a
    # borrow cannot be null, so those three assertions are gone.

    var local_start = 0

    comptime if RIEMANN_DEBUG:
        if local_start == 0:
            var tsize_val = tupleMultiplicity.size(nHits)
            print(
                tsize_val, "Ntuple of size", nHits, "for", hitsInFit, "hits to fit"
            )

    var local_idx: Int = local_start
    var nt = Int(Rfit.maxNumberOfConcurrentFits())
    while local_idx < nt:
        var tuple_idx = local_idx + Int(offset)
        if tuple_idx >= Int(tupleMultiplicity.size(nHits)):
            break

        var tkid = UInt32(tupleMultiplicity.begin(nHits)[tuple_idx])
        debug_assert(tkid < foundNtuplets.nbins())
        debug_assert(foundNtuplets.size(tkid) == nHits)

        var hits = Rfit.Map3xNd[N](phits[local_idx:])
        var fast_fit = Rfit.Map4d(pfast_fit[local_idx:])
        var hits_ge = Rfit.Map6xNf[N](phits_ge[local_idx:])

        var hitId = foundNtuplets.begin(tkid)
        var i: UInt32 = 0
        while i < hitsInFit:
            var idx = Int(i)
            var hit = Int(hitId[idx])
            var ge = InlineArray[Float32, 6](fill=0)
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
            i += 1

        RiemannFit.Fast_fit(hits, fast_fit)

        debug_assert(fast_fit[0] == fast_fit[0])
        debug_assert(fast_fit[1] == fast_fit[1])
        debug_assert(fast_fit[2] == fast_fit[2])
        debug_assert(fast_fit[3] == fast_fit[3])

        local_idx += 1

def kernelCircleFit[N: Int](
    tupleMultiplicity: CAConstants.TupleMultiplicity,
    nHits: UInt32,
    B: Float64,
    phits: Span[mut=True, Float64, _],
    phits_ge: Span[mut=True, Float32, _],
    pfast_fit_input: Span[mut=True, Float64, _],
    circle_fit: Span[mut=True, CircleFit, _],
    offset: UInt32,
):
    debug_assert(N <= Int(nHits))

    var local_start = 0
    var local_idx: Int = local_start
    var nt = Int(Rfit.maxNumberOfConcurrentFits())
    while local_idx < nt:
        var tuple_idx = local_idx + Int(offset)
        if tuple_idx >= Int(tupleMultiplicity.size(nHits)):
            break

        var hits = Rfit.Map3xNd[N](phits[local_idx:])
        var fast_fit = Rfit.Map4d(pfast_fit_input[local_idx:])
        var hits_ge = Rfit.Map6xNf[N](phits_ge[local_idx:])

        var rad = FitRfit.VectorNd[N]()
        var i: Int = 0
        while i < N:
            var x = hits[0, i]
            var y = hits[1, i]
            rad[i] = math.sqrt(x * x + y * y)
            i += 1

        var hits_cov = FitRfit.Matrix2Nd[N].Zero()
        FitRfit.loadCovariance2D(hits_ge, hits_cov)

        circle_fit[local_idx] = RiemannFit.Circle_fit(
            hits.block[2, N](0, 0),
            hits_cov,
            fast_fit,
            rad,
            B,
            True,
        )

        comptime if RIEMANN_DEBUG:
            pass
            # let tkid = tupleMultiplicity.begin(nHits)[tuple_idx]
            # print(
            #     f"kernelCircleFit circle.par(0,1,2): {tkid} {circle_fit[local_idx].par[0]},{circle_fit[local_idx].par[1]},{circle_fit[local_idx].par[2]}",
            # )

        local_idx += 1


def kernelLineFit[N: Int](
    tupleMultiplicity: CAConstants.TupleMultiplicity,
    nHits: UInt32,
    B: Float64,
    mut results: OutputSoA,
    phits: Span[mut=True, Float64, _],
    phits_ge: Span[mut=True, Float32, _],
    pfast_fit_input: Span[mut=True, Float64, _],
    circle_fit: Span[mut=True, CircleFit, _],
    offset: UInt32,
):
    debug_assert(N <= Int(nHits))

    var local_start = 0
    var local_idx: Int = local_start
    var nt = Int(Rfit.maxNumberOfConcurrentFits())
    var tuples_for_size = Int(tupleMultiplicity.size(nHits))
    while local_idx < nt:
        var tuple_idx = local_idx + Int(offset)
        if tuple_idx >= tuples_for_size:
            break

        var tkid = tupleMultiplicity.begin(nHits)[tuple_idx]

        var hits = Rfit.Map3xNd[N](phits[local_idx:])
        var fast_fit = Rfit.Map4d(pfast_fit_input[local_idx:])
        var hits_ge = Rfit.Map6xNf[N](phits_ge[local_idx:])

        ref line_fit = RiemannFit.Line_fit(
            hits,
            hits_ge,
            circle_fit[local_idx],
            fast_fit,
            B,
            True,
        )

        FitRfit.fromCircleToPerigee(circle_fit[local_idx])

        var track_idx = Int(tkid)
        var cp_buf = InlineArray[Scalar[DType.float64], 3](uninitialized=True)
        var ccov_buf = InlineArray[Scalar[DType.float64], 9](uninitialized=True)
        var lp_buf = InlineArray[Scalar[DType.float64], 2](uninitialized=True)
        var lcov_buf = InlineArray[Scalar[DType.float64], 4](uninitialized=True)
        results.stateAtBS.copyFromCircle(
            to_layout_tensor(circle_fit[local_idx].par, cp_buf),
            to_layout_tensor(circle_fit[local_idx].cov, ccov_buf),
            to_layout_tensor(line_fit.par, lp_buf),
            to_layout_tensor(line_fit.cov, lcov_buf),
            Float32(1.0 / B),
            Int32(track_idx),
        )
        results.pt[track_idx] = Float32(B) / Float32(
            abs(circle_fit[local_idx].par[2])
        )
        results.eta[track_idx] = Float32(math.asinh(line_fit.par[0]))
        var chi2 = Float64(circle_fit[local_idx].chi2) + line_fit.chi2
        results.chi2[track_idx] = Float32(
            chi2 / Float64(2 * N - 5)
        )

        comptime if RIEMANN_DEBUG:
            print(
                "kernelLineFit size", N, "for", nHits,
                "hits circle.par(0,1,2):", tkid,
                circle_fit[local_idx].par[0], ",", circle_fit[local_idx].par[1], ",", circle_fit[local_idx].par[2],
            )
            print(
                "kernelLineFit line.par(0,1):", tkid,
                line_fit.par[0], ",", line_fit.par[1],
            )
            print(
                "kernelLineFit chi2 cov", circle_fit[local_idx].chi2, "/", line_fit.chi2,
                circle_fit[local_idx].cov[0, 0], ",", circle_fit[local_idx].cov[1, 1], ",", circle_fit[local_idx].cov[2, 2],
                ",", line_fit.cov[0, 0], ",", line_fit.cov[1, 1],
            )

        local_idx += 1
