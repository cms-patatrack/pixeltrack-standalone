from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import PixelTrackHeterogeneous
import MojoSerial.plugin_PixelTriplets.CAConstants as CAConstants
import MojoSerial.plugin_PixelTriplets.FitResults as FitResults
from MojoSerial.CUDACore.CUDACompat import CUDAStreamType
from MojoSerial.MojoBridge.Matrix import Matrix, Map
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import PixelTrack as pixelTrack
from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from MojoSerial.plugin_PixelTriplets.FitUtils import Rfit as FitUtilsRfit
from MojoSerial.plugin_PixelTriplets.RiemannFitOnGPU import (
    kernelFastFit,
    kernelCircleFit,
    kernelLineFit,
)
from MojoSerial.plugin_PixelTriplets.BrokenLineFitOnGPU import (
    kernelBLFastFitAndFit,
)

comptime cudaStream_t = CUDAStreamType

struct Rfit:
    # in case of memory issue can be made smaller
    @staticmethod
    @parameter
    def maxNumberOfConcurrentFits() -> UInt32:
        return CAConstants.maxNumberOfTuples()

    @staticmethod
    @parameter
    def stride() -> UInt32:
        return Rfit.maxNumberOfConcurrentFits()

    # Each Map alias carries the backing buffer's origin; it is inferred from
    # the Span passed at the call site, so use sites still read `Map3xNd[N](s)`.
    comptime Matrix3x4d = Matrix[DType.float64, 3, 4]
    comptime Map3x4d[origin: MutOrigin] = Map[
        DType.float64,
        3,
        4,
        origin,
        Int(Self.stride()),
    ]
    comptime Matrix6x4f = Matrix[DType.float32, 6, 4]
    comptime Map6x4f[origin: MutOrigin] = Map[
        DType.float32,
        6,
        4,
        origin,
        Int(Self.stride()),
    ]

    # hits
    comptime Matrix3xNd[N: Int] = Matrix[DType.float64, 3, N]
    comptime Map3xNd[N: Int, origin: MutOrigin] = Map[
        DType.float64,
        3,
        N,
        origin,
        Int(Self.stride()),
    ]

    # errors
    comptime Matrix6xNf[N: Int] = Matrix[DType.float32, 6, N]
    comptime Map6xNf[N: Int, origin: MutOrigin] = Map[
        DType.float32,
        6,
        N,
        origin,
        Int(Self.stride()),
    ]

    # fast fit
    comptime Vector4d = Matrix[DType.float64, 4, 1]
    comptime Map4d[origin: MutOrigin] = Map[
        DType.float64, 4, 1, origin, Int(Self.stride())
    ]

    comptime Matrix3d = Matrix[DType.float64, 3, 3]
    comptime line_fit = FitUtilsRfit.line_fit


# C++: HelixFitOnGPU is a plain (non-template) class -- ported directly as a
# struct with real methods, same fix as CAHitNtupletGeneratorKernelsCPU
# (a `trait` with `var` fields is invalid Mojo: "fields in traits are not
# supported yet").
struct HelixFitOnGPU:
    comptime HitsView = TrackingRecHit2DHeterogeneous

    comptime Tuples = pixelTrack.HitContainer
    comptime OutputSoA = pixelTrack.TrackSoA

    comptime TupleMultiplicity = CAConstants.TupleMultiplicity

    comptime maxNumberOfConcurrentFits_ = Rfit.maxNumberOfConcurrentFits()

    # C++ stores three pointers forwarded by allocateOnGPU; passed to the launch
    # methods instead, so allocateOnGPU/deallocateOnGPU are gone. See §16.
    var bField_: Float32
    var fit5as4_: Bool

    def __init__(out self, bf: Float32, fit5as4: Bool):
        self.bField_ = bf
        self.fit5as4_ = fit5as4

    def setBField(mut self, bField: Float64):
        self.bField_ = Float32(bField)

    # C++ only declares these (GPU-kernel-launch wrappers); no .cc anywhere in
    # the serial backend ever defines them -- the CPU backend always calls
    # the *OnCPU variants below instead (see CAHitNtupletGeneratorOnGPU.cc).
    def launchRiemannKernels(
        self,
        hv: Self.HitsView,
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
        cudaStream: cudaStream_t
    ):
        pass

    def launchBrokenLineKernels(
        self,
        hv: Self.HitsView,
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
        cudaStream: cudaStream_t
    ):
        pass

    # C++: HelixFitOnGPU::launchRiemannKernelsOnCPU (RiemannFitOnGPU.cc)
    # C++'s `tuples_d = &soa->hitIndices` aliases `outputSoa_d = soa`; derived
    # from `outputSoa` here instead.
    def launchRiemannKernelsOnCPU(
        self,
        hv: Self.HitsView,
        cpeParams: PixelCPEFast,
        tupleMultiplicity: Self.TupleMultiplicity,
        mut outputSoa: Self.OutputSoA,
        nhits: UInt32,
        maxNumberOfTuples: UInt32
    ) raises:
        var nConcurrent = Int(Self.maxNumberOfConcurrentFits_)
        # Fit internals -- C++ sizes these as
        # maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double)
        # etc.; ported directly as element counts (List[T] is this port's
        # unique_ptr<T[]> equivalent) instead of raw byte arithmetic.
        var hitsGPU_ = List[Float64](length=nConcurrent * 3 * 4, fill=0.0)
        var hits_geGPU_ = List[Float32](length=nConcurrent * 6 * 4, fill=0.0)
        var fast_fit_resultsGPU_ = List[Float64](length=nConcurrent * 4, fill=0.0)
        var circle_fit_resultsGPU_ = List[FitUtilsRfit.circle_fit](
            length=nConcurrent,
            fill=FitUtilsRfit.circle_fit(
                FitUtilsRfit.Vector3d(), FitUtilsRfit.Matrix3d(), 0, 0
            ),
        )

        var phits = Span(hitsGPU_)
        var phits_ge = Span(hits_geGPU_)
        var pfast_fit = Span(fast_fit_resultsGPU_)
        var pcircle_fit = Span(circle_fit_resultsGPU_)

        var offset: UInt32 = 0
        while offset < maxNumberOfTuples:
            # triplets
            kernelFastFit[3](
                outputSoa.hitIndices, tupleMultiplicity, 3, hv, cpeParams,
                phits, phits_ge, pfast_fit, offset,
            )
            kernelCircleFit[3](
                tupleMultiplicity, 3, Float64(self.bField_),
                phits, phits_ge, pfast_fit, pcircle_fit, offset,
            )
            kernelLineFit[3](
                tupleMultiplicity, 3, Float64(self.bField_),
                outputSoa, phits, phits_ge, pfast_fit, pcircle_fit, offset,
            )

            # quads
            kernelFastFit[4](
                outputSoa.hitIndices, tupleMultiplicity, 4, hv, cpeParams,
                phits, phits_ge, pfast_fit, offset,
            )
            kernelCircleFit[4](
                tupleMultiplicity, 4, Float64(self.bField_),
                phits, phits_ge, pfast_fit, pcircle_fit, offset,
            )
            kernelLineFit[4](
                tupleMultiplicity, 4, Float64(self.bField_),
                outputSoa, phits, phits_ge, pfast_fit, pcircle_fit, offset,
            )

            if self.fit5as4_:
                # penta (only first 4)
                kernelFastFit[4](
                    outputSoa.hitIndices, tupleMultiplicity, 5, hv, cpeParams,
                    phits, phits_ge, pfast_fit, offset,
                )
                kernelCircleFit[4](
                    tupleMultiplicity, 5, Float64(self.bField_),
                    phits, phits_ge, pfast_fit, pcircle_fit, offset,
                )
                kernelLineFit[4](
                    tupleMultiplicity, 5, Float64(self.bField_),
                    outputSoa, phits, phits_ge, pfast_fit, pcircle_fit, offset,
                )
            else:
                # penta (all 5)
                kernelFastFit[5](
                    outputSoa.hitIndices, tupleMultiplicity, 5, hv, cpeParams,
                    phits, phits_ge, pfast_fit, offset,
                )
                kernelCircleFit[5](
                    tupleMultiplicity, 5, Float64(self.bField_),
                    phits, phits_ge, pfast_fit, pcircle_fit, offset,
                )
                kernelLineFit[5](
                    tupleMultiplicity, 5, Float64(self.bField_),
                    outputSoa, phits, phits_ge, pfast_fit, pcircle_fit, offset,
                )

            offset += Self.maxNumberOfConcurrentFits_

    # C++: HelixFitOnGPU::launchBrokenLineKernelsOnCPU (BrokenLineFitOnGPU.cc)
    def launchBrokenLineKernelsOnCPU(
        self,
        hv: Self.HitsView,
        cpeParams: PixelCPEFast,
        tupleMultiplicity: Self.TupleMultiplicity,
        mut outputSoa: Self.OutputSoA,
        nhits: UInt32,
        maxNumberOfTuples: UInt32
    ) raises:
        # the GPU kernels process the n-tuplets in chunks of maxNumberOfConcurrentFits_, up to maxNumberOfTuples
        var nConcurrent = Self.maxNumberOfConcurrentFits_
        var maxTuples = (maxNumberOfTuples + nConcurrent - 1) // nConcurrent * nConcurrent
        var B = Float64(self.bField_)

        # fit triplets
        kernelBLFastFitAndFit[3](
            tupleMultiplicity, hv, cpeParams, B, outputSoa, 3, maxTuples
        )
        # fit quads
        kernelBLFastFitAndFit[4](
            tupleMultiplicity, hv, cpeParams, B, outputSoa, 4, maxTuples
        )
        if self.fit5as4_:
            # fit penta (only first 4)
            kernelBLFastFitAndFit[4](
                tupleMultiplicity, hv, cpeParams, B, outputSoa, 5, maxTuples
            )
        else:
            # fit penta (all 5)
            kernelBLFastFitAndFit[5](
                tupleMultiplicity, hv, cpeParams, B, outputSoa, 5, maxTuples
            )
