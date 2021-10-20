#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

#include "KokkosDataFormats/PixelTrackKokkos.h"
#include "../GPUCACell.h"

// #define DUMP_GPU_TK_TUPLES

namespace cAHitNtupletGenerator {

  // counters
  struct Counters {
    unsigned long long nEvents;
    unsigned long long nHits;
    unsigned long long nCells;
    unsigned long long nTuples;
    unsigned long long nFitTracks;
    unsigned long long nGoodTracks;
    unsigned long long nUsedHits;
    unsigned long long nDupHits;
    unsigned long long nKilledCells;
    unsigned long long nEmptyCells;
    unsigned long long nZeroTrackCells;
  };

  using HitsView = TrackingRecHit2DSOAView;
  using HitsOnGPU = TrackingRecHit2DSOAView;

  using HitToTuple = CAConstants::HitToTuple;
  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

  struct QualityCuts {
    // chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    float chi2Coeff[4];
    float chi2MaxPt;  // GeV
    float chi2Scale;

    struct region {
      float maxTip;  // cm
      float minPt;   // GeV
      float maxZip;  // cm
    };

    region triplet;
    region quadruplet;
  };

  // params
  struct Params {
    Params(bool onGPU,
           uint32_t minHitsPerNtuplet,
           uint32_t maxNumberOfDoublets,
           bool useRiemannFit,
           bool fit5as4,
           bool includeJumpingForwardDoublets,
           bool earlyFishbone,
           bool lateFishbone,
           bool idealConditions,
           bool doStats,
           bool doClusterCut,
           bool doZ0Cut,
           bool doPtCut,
           float ptmin,
           float CAThetaCutBarrel,
           float CAThetaCutForward,
           float hardCurvCut,
           float dcaCutInnerTriplet,
           float dcaCutOuterTriplet,
           QualityCuts const& cuts)
        : onGPU_(onGPU),
          minHitsPerNtuplet_(minHitsPerNtuplet),
          maxNumberOfDoublets_(maxNumberOfDoublets),
          useRiemannFit_(useRiemannFit),
          fit5as4_(fit5as4),
          includeJumpingForwardDoublets_(includeJumpingForwardDoublets),
          earlyFishbone_(earlyFishbone),
          lateFishbone_(lateFishbone),
          idealConditions_(idealConditions),
          doStats_(doStats),
          doClusterCut_(doClusterCut),
          doZ0Cut_(doZ0Cut),
          doPtCut_(doPtCut),
          ptmin_(ptmin),
          CAThetaCutBarrel_(CAThetaCutBarrel),
          CAThetaCutForward_(CAThetaCutForward),
          hardCurvCut_(hardCurvCut),
          dcaCutInnerTriplet_(dcaCutInnerTriplet),
          dcaCutOuterTriplet_(dcaCutOuterTriplet),
          cuts_(cuts) {}

    const bool onGPU_;
    const uint32_t minHitsPerNtuplet_;
    const uint32_t maxNumberOfDoublets_;
    const bool useRiemannFit_;
    const bool fit5as4_;
    const bool includeJumpingForwardDoublets_;
    const bool earlyFishbone_;
    const bool lateFishbone_;
    const bool idealConditions_;
    const bool doStats_;
    const bool doClusterCut_;
    const bool doZ0Cut_;
    const bool doPtCut_;
    const float ptmin_;
    const float CAThetaCutBarrel_;
    const float CAThetaCutForward_;
    const float hardCurvCut_;
    const float dcaCutInnerTriplet_;
    const float dcaCutOuterTriplet_;

    // quality cuts
    QualityCuts cuts_{// polynomial coefficients for the pT-dependent chi2 cut
                      {0.68177776, 0.74609577, -0.08035491, 0.00315399},
                      // max pT used to determine the chi2 cut
                      10.,
                      // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                      30.,
                      // regional cuts for triplets
                      {
                          0.3,  // |Tip| < 0.3 cm
                          0.5,  // pT > 0.5 GeV
                          12.0  // |Zip| < 12.0 cm
                      },
                      // regional cuts for quadruplets
                      {
                          0.5,  // |Tip| < 0.5 cm
                          0.3,  // pT > 0.3 GeV
                          12.0  // |Zip| < 12.0 cm
                      }};

  };  // Params

}  // namespace cAHitNtupletGenerator

namespace KOKKOS_NAMESPACE {
  class CAHitNtupletGeneratorKernels {
  public:
    using QualityCuts = cAHitNtupletGenerator::QualityCuts;
    using Params = cAHitNtupletGenerator::Params;
    using Counters = cAHitNtupletGenerator::Counters;

    using HitsView = TrackingRecHit2DSOAView;
    using HitsOnGPU = TrackingRecHit2DSOAView;
    using HitsOnCPU = TrackingRecHit2DKokkos<KokkosDeviceMemSpace>;

    using HitToTuple = CAConstants::HitToTuple;
    using TupleMultiplicity = CAConstants::TupleMultiplicity;

    using Quality = pixelTrack::Quality;
    using TkSoA = pixelTrack::TrackSoA;
    using HitContainer = pixelTrack::HitContainer;

    CAHitNtupletGeneratorKernels(Params const& params) : m_params(params) {}
    ~CAHitNtupletGeneratorKernels() = default;

    Kokkos::View<TupleMultiplicity const, KokkosDeviceMemSpace, Restrict> tupleMultiplicity() const {
      return device_tupleMultiplicity_;
    }

    void launchKernels(HitsOnCPU const& hh,
                       const Kokkos::View<TkSoA, KokkosDeviceMemSpace, Restrict>& tuples_d,
                       KokkosExecSpace const& execSpace);

    void classifyTuples(HitsOnCPU const& hh,
                        const Kokkos::View<TkSoA, KokkosDeviceMemSpace, Restrict>& tuples_d,
                        KokkosExecSpace const& execSpace);

    void fillHitDetIndices(HitsView const* hv,
                           const Kokkos::View<TkSoA, KokkosDeviceMemSpace, Restrict>& tuples_d,
                           KokkosExecSpace const& execSpace);

    void buildDoublets(HitsOnCPU const& hh, KokkosExecSpace const& execSpace);

    void allocateOnGPU(KokkosExecSpace const& execSpace);

    static void printCounters(const Kokkos::View<Counters const, KokkosDeviceMemSpace, Restrict>& counters);

    Counters* counters_ = nullptr;

  private:
    Kokkos::View<CAConstants::CellNeighborsVector, KokkosDeviceMemSpace, Restrict> device_theCellNeighbors_;
    Kokkos::View<CAConstants::CellNeighbors*, KokkosDeviceMemSpace, Restrict> device_theCellNeighborsContainer_;
    Kokkos::View<CAConstants::CellTracksVector, KokkosDeviceMemSpace, Restrict> device_theCellTracks_;
    Kokkos::View<CAConstants::CellTracks*, KokkosDeviceMemSpace, Restrict> device_theCellTracksContainer_;

    Kokkos::View<GPUCACell*, KokkosDeviceMemSpace, Restrict> device_theCells_;
    Kokkos::View<GPUCACell::OuterHitOfCell*, KokkosDeviceMemSpace, Restrict> device_isOuterHitOfCell_;
    Kokkos::View<uint32_t, KokkosDeviceMemSpace, Restrict> device_nCells_;

    Kokkos::View<HitToTuple, KokkosDeviceMemSpace, Restrict> device_hitToTuple_;

    Kokkos::View<cms::kokkos::AtomicPairCounter, KokkosDeviceMemSpace, Restrict> device_hitToTuple_apc_;

    Kokkos::View<cms::kokkos::AtomicPairCounter, KokkosDeviceMemSpace, Restrict> device_hitTuple_apc_;

    Kokkos::View<TupleMultiplicity, KokkosDeviceMemSpace, Restrict> device_tupleMultiplicity_;

    Kokkos::View<uint8_t*, KokkosDeviceMemSpace, Restrict> device_tmws_;

    //Kokkos::View<cms::kokkos::AtomicPairCounter::c_type*, KokkosDeviceMemSpace> device_storage_;

    // params
    Params const& m_params;
  };

}  // namespace KOKKOS_NAMESPACE

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
