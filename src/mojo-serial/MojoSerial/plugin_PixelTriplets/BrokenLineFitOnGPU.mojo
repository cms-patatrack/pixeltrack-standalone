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


# C++ reads `hhp->cpeParams()`; that field went with the view (§11).
def kernelBLFastFit[N: Int](
    foundNtuplets: Tuples,
    tupleMultiplicity: CAConstants.TupleMultiplicity,
    hhp: HitsOnGPU,
    cpeParams: PixelCPEFast,
    phits: Span[mut=True, Float64, _],
    phits_ge: Span[mut=True, Float32, _],
    pfast_fit: Span[mut=True, Float64, _],
    nHits: UInt32,
    offset: UInt32,
):
    var hitsInFit: UInt32 = UInt32(N)
    debug_assert(hitsInFit <= nHits)
    # C++ asserts hhp/pfast_fit/foundNtuplets/tupleMultiplicity are non-null;
    # a borrow cannot be null, so those four assertions are gone.

    var local_start = 0

    comptime if BROKENLINE_DEBUG:
        if local_start == 0:
            var nbins_val = foundNtuplets.nbins()
            print(nbins_val, "total Ntuple")
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

        var dump: Bool = False

        comptime if BL_DUMP_HITS:
            var done = Atomic[DType.int64](0)
            dump = (
                foundNtuplets.size(tkid) == 5 and done.fetch_add(1) == 0
            )

        var hitId = foundNtuplets.begin(tkid)
        var i: UInt32 = 0
        while i < hitsInFit:
            var idx = Int(i)
            var hit = Int(hitId[idx])
            var ge = InlineArray[Float32, 6](uninitialized=True)
            cpeParams.detParams(Int(hhp.detectorIndex(hit))).frame.toGlobal(
                hhp.xerrLocal(hit),
                0,
                hhp.yerrLocal(hit),
                Span(ge),
            )

            comptime if BL_DUMP_HITS:
                if dump:
                    var det_idx = hhp.detectorIndex(hit)
                    var xg = hhp.xGlobal(hit)
                    var yg = hhp.yGlobal(hit)
                    var zg = hhp.zGlobal(hit)
                    print(
                        "Hit global:", tkid, ":", det_idx, "hits.col(", i, ") <<",
                        xg, ",", yg, ",", zg,
                    )
                    print(
                        "Error:", tkid, ":", det_idx, " hits_ge.col(", i, ") <<",
                        ge[0], ",", ge[1], ",", ge[2], ",", ge[3], ",", ge[4], ",", ge[5],
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

        BrokenLine.BL_Fast_fit(hits, fast_fit)


        debug_assert(fast_fit[0] == fast_fit[0])
        debug_assert(fast_fit[1] == fast_fit[1])
        debug_assert(fast_fit[2] == fast_fit[2])
        debug_assert(fast_fit[3] == fast_fit[3])

        local_idx += 1


def kernelBLFit[N: Int](
    tupleMultiplicity: CAConstants.TupleMultiplicity,
    B: Float64,
    mut results: OutputSoA,
    phits: Span[mut=True, Float64, _],
    phits_ge: Span[mut=True, Float32, _],
    pfast_fit: Span[mut=True, Float64, _],
    nHits: UInt32,
    offset: UInt32,
):
    debug_assert(N <= Int(nHits))
    # C++ asserts results/pfast_fit are non-null; borrows cannot be null.

    var local_start = 0
    var local_idx: Int = local_start
    var nt = Int(Rfit.maxNumberOfConcurrentFits())
    var tuples_for_size = Int(tupleMultiplicity.size(nHits))
    while local_idx < nt:
        var tuple_idx = local_idx + Int(offset)
        if tuple_idx >= tuples_for_size:
            break

        var tkid = UInt32(tupleMultiplicity.begin(nHits)[tuple_idx])

        var hits = Rfit.Map3xNd[N](phits[local_idx:])
        var fast_fit = Rfit.Map4d(pfast_fit[local_idx:])
        var hits_ge = Rfit.Map6xNf[N](phits_ge[local_idx:])

        var data = BrokenLine.PreparedBrokenLineData[N]()
        var Jacob = Rfit.Matrix3d(uninitialized=True)
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
            print(
                "kernelBLHits line.par(0,1):", tkid,
                line.par[0], ",", line.par[1],
            )
            print(
                "kernelBLHits chi2 cov", circle.chi2, "/", line.chi2,
                " ", circle.cov[0, 0], ",", circle.cov[1, 1], ",", circle.cov[2, 2],
                ",", line.cov[0, 0], ",", line.cov[1, 1],
            )

        local_idx += 1
