#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "GPUCACell.h"

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

  using HitsView = ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHit2DSOAView;
  using HitsOnGPU = ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHit2DSOAView;

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

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CAHitNtupletGeneratorKernels {
  public:
    using QualityCuts = cAHitNtupletGenerator::QualityCuts;
    using Params = cAHitNtupletGenerator::Params;
    using Counters = cAHitNtupletGenerator::Counters;

    using HitsView = TrackingRecHit2DSOAView;
    using HitsOnGPU = TrackingRecHit2DSOAView;
    using HitsOnCPU = TrackingRecHit2DAlpaka;

    using HitToTuple = CAConstants::HitToTuple;
    using TupleMultiplicity = CAConstants::TupleMultiplicity;

    using Quality = pixelTrack::Quality;
    using TkSoA = pixelTrack::TrackSoA;
    using HitContainer = pixelTrack::HitContainer;

    CAHitNtupletGeneratorKernels(Params const& params, uint32_t nhits)
        : m_params(params),
          //////////////////////////////////////////////////////////
          // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
          //////////////////////////////////////////////////////////
          counters_{cms::alpakatools::allocDeviceBuf<Counters>(1u)},

          device_hitToTuple_{cms::alpakatools::allocDeviceBuf<HitToTuple>(1u)},
          device_tupleMultiplicity_{cms::alpakatools::allocDeviceBuf<TupleMultiplicity>(1u)},

          device_theCells_{cms::alpakatools::allocDeviceBuf<GPUCACell>(params.maxNumberOfDoublets_)},
          // in principle we can use "nhits" to heuristically dimension the workspace...
          device_isOuterHitOfCell_{cms::alpakatools::allocDeviceBuf<GPUCACell::OuterHitOfCell>(std::max(1U, nhits))},

          device_theCellNeighbors_{cms::alpakatools::allocDeviceBuf<CAConstants::CellNeighborsVector>(1u)},
          device_theCellTracks_{cms::alpakatools::allocDeviceBuf<CAConstants::CellTracksVector>(1u)},

          //cellStorage_{cms::alpakatools::allocDeviceBuf<unsigned char>(CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors) + CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellTracks))},
          device_theCellNeighborsContainer_{
              cms::alpakatools::allocDeviceBuf<CAConstants::CellNeighbors>(CAConstants::maxNumOfActiveDoublets())},
          device_theCellTracksContainer_{
              cms::alpakatools::allocDeviceBuf<CAConstants::CellTracks>(CAConstants::maxNumOfActiveDoublets())},

          //device_storage_{cms::alpakatools::allocDeviceBuf<cms::cuda::AtomicPairCounter::c_type>(3u)},
          //device_hitTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get()},
          //device_hitToTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get() + 1;
          //device_nCells_ = (uint32_t*)(device_storage_.get() + 2)},
          device_hitTuple_apc_{cms::alpakatools::allocDeviceBuf<cms::alpakatools::AtomicPairCounter>(1u)},
          device_hitToTuple_apc_{cms::alpakatools::allocDeviceBuf<cms::alpakatools::AtomicPairCounter>(1u)},
          device_nCells_{cms::alpakatools::allocDeviceBuf<uint32_t>(1u)} {
      Queue queue(device);

      alpaka::memset(queue, counters_, 0, 1u);

      alpaka::memset(queue, device_nCells_, 0, 1u);

      launchZero(alpaka::getPtrNative(device_tupleMultiplicity_), queue);
      launchZero(alpaka::getPtrNative(device_hitToTuple_), queue);

      // we may wish to keep it in the edm...
      alpaka::wait(queue);
    }

    ~CAHitNtupletGeneratorKernels() = default;

    TupleMultiplicity const* tupleMultiplicity() const { return alpaka::getPtrNative(device_tupleMultiplicity_); }

    void launchKernels(HitsOnCPU const& hh, TkSoA* tuples_d, Queue& queue);

    void classifyTuples(HitsOnCPU const& hh, TkSoA* tuples_d, Queue& queue);

    void fillHitDetIndices(HitsView const* hv, TkSoA* tuples_d, Queue& queue);

    void buildDoublets(HitsOnCPU const& hh, Queue& queue);
    void cleanup(Queue& queue);

    void printCounters(Queue& queue);
    //Counters* counters_ = nullptr;

  private:
    // params
    Params const& m_params;

    AlpakaDeviceBuf<Counters> counters_;  // NB: Counters: In legacy, sum of the stats of all events.
    // Here instead, these stats are per event.
    // Does not matter much, as the stats are desactivated by default anyway, and are for debug only
    // (stats are not stored eventually, no interference with any result).
    // For debug, better to be able to see info per event that just a sum.

    // workspace
    AlpakaDeviceBuf<HitToTuple> device_hitToTuple_;
    AlpakaDeviceBuf<TupleMultiplicity> device_tupleMultiplicity_;

    AlpakaDeviceBuf<GPUCACell> device_theCells_;  // NB: In legacy, was allocated inside buildDoublets.
    AlpakaDeviceBuf<GPUCACell::OuterHitOfCell>
        device_isOuterHitOfCell_;  // NB: In legacy, was allocated inside buildDoublets.

    AlpakaDeviceBuf<CAConstants::CellNeighborsVector> device_theCellNeighbors_;
    AlpakaDeviceBuf<CAConstants::CellTracksVector> device_theCellTracks_;

    // AlpakaDeviceBuf<unsigned char> cellStorage_; // NB: In legacy, was allocated inside buildDoublets.
    // NB: Here, data from cellstorage_ (legacy) directly owned by the following:
    AlpakaDeviceBuf<CAConstants::CellNeighbors> device_theCellNeighborsContainer_;  // Was non-owning in legacy!
    AlpakaDeviceBuf<CAConstants::CellTracks> device_theCellTracksContainer_;        // Was non-owning in legacy!

    // AlpakaDeviceBuf<cms::alpakatools::AtomicPairCounter::c_type> device_storage_; // NB: In legacy
    // NB: Here, data from device_storage_ (legacy) directly owned by the following:
    AlpakaDeviceBuf<cms::alpakatools::AtomicPairCounter> device_hitTuple_apc_;    // Was non-owning in legacy!
    AlpakaDeviceBuf<cms::alpakatools::AtomicPairCounter> device_hitToTuple_apc_;  // Was non-owning in legacy!
    AlpakaDeviceBuf<uint32_t> device_nCells_;                                     // Was non-owning in legacy!
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
