#ifndef plugin_PixelTriplets_alpaka_CAHitNtupletGeneratorKernels_h
#define plugin_PixelTriplets_alpaka_CAHitNtupletGeneratorKernels_h

#include <algorithm>

#include "AlpakaCore/alpakaMemory.h"
#include "AlpakaDataFormats/alpaka/PixelTrackAlpaka.h"

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

  using HitsView = TrackingRecHit2DSoAView;
  using HitsOnGPU = TrackingRecHit2DSoAView;

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

    using HitsView = TrackingRecHit2DSoAView;
    using HitsOnGPU = TrackingRecHit2DSoAView;
    using HitsOnCPU = TrackingRecHit2DAlpaka;

    using HitToTuple = CAConstants::HitToTuple;
    using TupleMultiplicity = CAConstants::TupleMultiplicity;

    using Quality = pixelTrack::Quality;
    using TkSoA = pixelTrack::TrackSoA;
    using HitContainer = pixelTrack::HitContainer;

    CAHitNtupletGeneratorKernels(Params const& params, uint32_t nhits, Queue& queue)
        : m_params(params),
          //////////////////////////////////////////////////////////
          // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
          //////////////////////////////////////////////////////////
          counters_{cms::alpakatools::make_device_buffer<Counters>(queue)},

          // workspace
          device_hitToTuple_{cms::alpakatools::make_device_buffer<HitToTuple>(queue)},
          device_tupleMultiplicity_{cms::alpakatools::make_device_buffer<TupleMultiplicity>(queue)},

          // NB: In legacy, device_theCells_ and device_isOuterHitOfCell_ were allocated inside buildDoublets
          device_theCells_{cms::alpakatools::make_device_buffer<GPUCACell[]>(queue, params.maxNumberOfDoublets_)},
          // in principle we can use "nhits" to heuristically dimension the workspace...
          device_isOuterHitOfCell_{
              cms::alpakatools::make_device_buffer<GPUCACell::OuterHitOfCell[]>(queue, std::max(1u, nhits))},

          device_theCellNeighbors_{cms::alpakatools::make_device_buffer<CAConstants::CellNeighborsVector>(queue)},
          device_theCellTracks_{cms::alpakatools::make_device_buffer<CAConstants::CellTracksVector>(queue)},
          // NB: In legacy, cellStorage_ was allocated inside buildDoublets
          cellStorage_{cms::alpakatools::make_device_buffer<unsigned char[]>(
              queue,
              CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors) +
                  CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellTracks))},
          device_theCellNeighborsContainer_{reinterpret_cast<GPUCACell::CellNeighbors*>(cellStorage_.data())},
          device_theCellTracksContainer_{reinterpret_cast<GPUCACell::CellTracks*>(
              cellStorage_.data() + CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors))},

          // NB: In legacy, device_storage_ was allocated inside allocateOnGPU
          device_storage_{
              cms::alpakatools::make_device_buffer<cms::alpakatools::AtomicPairCounter::c_type[]>(queue, 3u)},
          device_hitTuple_apc_{reinterpret_cast<cms::alpakatools::AtomicPairCounter*>(device_storage_.data())},
          device_hitToTuple_apc_{reinterpret_cast<cms::alpakatools::AtomicPairCounter*>(device_storage_.data() + 1)},
          device_nCells_{cms::alpakatools::make_device_view(alpaka::getDev(queue),
                                                            *reinterpret_cast<uint32_t*>(device_storage_.data() + 2))} {
      alpaka::memset(queue, counters_, 0);
      alpaka::memset(queue, device_nCells_, 0);
      cms::alpakatools::launchZero<Acc1D>(device_tupleMultiplicity_.data(), queue);
      cms::alpakatools::launchZero<Acc1D>(device_hitToTuple_.data(), queue);
    }

    ~CAHitNtupletGeneratorKernels() = default;

    TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_.data(); }

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

    // NB: Counters: In legacy, sum of the stats of all events.
    // Here instead, these stats are per event.
    // Does not matter much, as the stats are desactivated by default anyway, and are for debug only
    // (stats are not stored eventually, no interference with any result).
    // For debug, better to be able to see info per event that just a sum.
    cms::alpakatools::device_buffer<Device, Counters> counters_;

    // workspace
    cms::alpakatools::device_buffer<Device, HitToTuple> device_hitToTuple_;
    cms::alpakatools::device_buffer<Device, TupleMultiplicity> device_tupleMultiplicity_;

    // NB: In legacy, device_theCells_ and device_isOuterHitOfCell_ were allocated inside buildDoublets
    cms::alpakatools::device_buffer<Device, GPUCACell[]> device_theCells_;
    cms::alpakatools::device_buffer<Device, GPUCACell::OuterHitOfCell[]> device_isOuterHitOfCell_;

    cms::alpakatools::device_buffer<Device, CAConstants::CellNeighborsVector> device_theCellNeighbors_;
    cms::alpakatools::device_buffer<Device, CAConstants::CellTracksVector> device_theCellTracks_;

    // NB: In legacy, cellStorage_ was allocated inside buildDoublets
    cms::alpakatools::device_buffer<Device, unsigned char[]> cellStorage_;
    CAConstants::CellNeighbors* device_theCellNeighborsContainer_;
    CAConstants::CellTracks* device_theCellTracksContainer_;

    // NB: In legacy, device_storage_ was allocated inside allocateOnGPU
    cms::alpakatools::device_buffer<Device, cms::alpakatools::AtomicPairCounter::c_type[]> device_storage_;
    cms::alpakatools::AtomicPairCounter* device_hitTuple_apc_;
    cms::alpakatools::AtomicPairCounter* device_hitToTuple_apc_;
    cms::alpakatools::device_view<Device, uint32_t> device_nCells_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_PixelTriplets_alpaka_CAHitNtupletGeneratorKernels_h
