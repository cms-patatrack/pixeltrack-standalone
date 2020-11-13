//
// Original Author: Felice Pantaleo, CERN
//

// #define NTUPLE_DEBUG

#include <cmath>
#include <cstdint>

#include "CondFormats/pixelCPEforGPU.h"
#include "KokkosCore/kokkosConfig.h"

#include "../CAConstants.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "../GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

namespace KOKKOS_NAMESPACE {
  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DKokkos<KokkosExecSpace>;

  using HitToTuple = CAConstants::HitToTuple;
  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

  // TeamPolicy instead of RangePolicy due to loops with different iteration counts
  KOKKOS_INLINE_FUNCTION void kernel_checkOverflows(
      HitContainer const *foundNtuplets,
      Kokkos::View<TupleMultiplicity, KokkosExecSpace> tupleMultiplicity,
      Kokkos::View<cms::kokkos::AtomicPairCounter, KokkosExecSpace> apc,
      Kokkos::View<GPUCACell *, KokkosExecSpace> cells,
      Kokkos::View<uint32_t, KokkosExecSpace> nCells,
      Kokkos::View<CAConstants::CellNeighborsVector, KokkosExecSpace> cellNeighbors,  // not used
      Kokkos::View<CAConstants::CellTracksVector, KokkosExecSpace> cellTracks,        // not used
      Kokkos::View<GPUCACell::OuterHitOfCell *, KokkosExecSpace> isOuterHitOfCell,
      uint32_t nHits,
      uint32_t maxNumberOfDoublets,
      cAHitNtupletGenerator::Counters *counters,
      const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
    const int teamRank = teamMember.team_rank();
    const int teamSize = teamMember.team_size();
    const int leagueRank = teamMember.league_rank();
    const int leagueSize = teamMember.league_size();

    auto first = teamRank + leagueRank * teamSize;

    auto &c = *counters;
    // counters once per event
    if (0 == first) {
      Kokkos::atomic_add<unsigned long long>(&c.nEvents, 1);
      Kokkos::atomic_add<unsigned long long>(&c.nHits, nHits);
      Kokkos::atomic_add<unsigned long long>(&c.nCells, nCells());
      Kokkos::atomic_add<unsigned long long>(&c.nTuples, apc().get().m);
      Kokkos::atomic_add<unsigned long long>(&c.nFitTracks, tupleMultiplicity().size());
    }

#ifdef NTUPLE_DEBUG
    if (0 == first) {
      printf("number of found cells %d, found tuples %d with total hits %d out of %d\n",
             nCells(),
             apc().get().m,
             apc().get().n,
             nHits);
      if (apc().get().m < CAConstants::maxNumberOfQuadruplets()) {
        assert(foundNtuplets->size(apc().get().m) == 0);
        assert(foundNtuplets->size() == apc().get().n);
      }
    }

    for (int idx = first, nt = foundNtuplets->nbins(); idx < nt; idx += leagueSize * teamSize) {
      if (foundNtuplets->size(idx) > 5)
        printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
      assert(foundNtuplets->size(idx) < 6);
      for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
        assert(*ih < nHits);
    }
#endif

    if (0 == first) {
      if (apc().get().m >= CAConstants::maxNumberOfQuadruplets())
        printf("Tuples overflow\n");
      if (nCells() >= maxNumberOfDoublets)
        printf("Cells overflow\n");
      if (cellNeighbors().full())
        printf("cellNeighbors overflow\n");
      if (cellTracks().full())
        printf("cellTracks overflow\n");
    }

    for (int idx = first, nt = nCells(); idx < nt; idx += leagueSize * teamSize) {
      auto const &thisCell = cells(idx);
      if (thisCell.outerNeighbors().full())  //++tooManyNeighbors[thisCell.theLayerPairId];
        printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
      if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
        printf("Tracks overflow %d in %d\n", idx, thisCell.theLayerPairId);
      if (thisCell.theDoubletId < 0)
        Kokkos::atomic_add<unsigned long long>(&c.nKilledCells, 1);
      if (0 == thisCell.theUsed)
        Kokkos::atomic_add<unsigned long long>(&c.nEmptyCells, 1);
      if (thisCell.tracks().empty())
        Kokkos::atomic_add<unsigned long long>(&c.nZeroTrackCells, 1);
    }

    for (int idx = first, nt = nHits; idx < nt; idx += leagueSize * teamSize) {
      if (isOuterHitOfCell(idx).full())  // ++tooManyOuterHitOfCell;
        printf("OuterHitOfCell overflow %d\n", idx);
    }
  }

  KOKKOS_INLINE_FUNCTION void kernel_fishboneCleaner(GPUCACell const *cells, Quality *quality, const size_t idx) {
    auto const &thisCell = cells[idx];
    if (thisCell.theDoubletId >= 0)
      return;

    constexpr auto bad = trackQuality::bad;

    for (auto it : thisCell.tracks())
      quality[it] = bad;
  }

  KOKKOS_INLINE_FUNCTION void kernel_earlyDuplicateRemover(Kokkos::View<GPUCACell *, KokkosExecSpace> cells,
                                                           HitContainer *foundNtuplets,
                                                           Quality *quality,
                                                           const size_t idx) {
    // constexpr auto bad = trackQuality::bad;
    constexpr auto dup = trackQuality::dup;
    // constexpr auto loose = trackQuality::loose;

    auto const &thisCell = cells(idx);

    if (thisCell.tracks().size() < 2)
      return;
    //if (0==thisCell.theUsed) continue;
    // if (thisCell.theDoubletId < 0) continue;

    uint32_t maxNh = 0;

    // find maxNh
    for (auto it : thisCell.tracks()) {
      auto nh = foundNtuplets->size(it);
      maxNh = std::max(nh, maxNh);
    }

    for (auto it : thisCell.tracks()) {
      if (foundNtuplets->size(it) != maxNh)
        quality[it] = dup;  //no race:  simple assignment of the same constant
    }
  }

  KOKKOS_INLINE_FUNCTION void kernel_fastDuplicateRemover(GPUCACell const *__restrict__ cells,
                                                          HitContainer const *__restrict__ foundNtuplets,
                                                          TkSoA *__restrict__ tracks,
                                                          const size_t idx) {
    constexpr auto bad = trackQuality::bad;
    constexpr auto dup = trackQuality::dup;
    constexpr auto loose = trackQuality::loose;

    auto const &thisCell = cells[idx];
    if (thisCell.tracks().size() < 2)
      return;
    // if (thisCell.theDoubletId < 0) return;

    float mc = 10000.f;
    uint16_t im = 60000;

    auto score = [&](auto it) {
      return std::abs(tracks->tip(it));  // tip
      // return tracks->chi2(it);  //chi2
    };

    // find min socre
    for (auto it : thisCell.tracks()) {
      if (tracks->quality(it) == loose && score(it) < mc) {
        mc = score(it);
        im = it;
      }
    }
    // mark all other duplicates
    for (auto it : thisCell.tracks()) {
      if (tracks->quality(it) != bad && it != im)
        tracks->quality(it) = dup;  //no race:  simple assignment of the same constant
    }
  }

  KOKKOS_INLINE_FUNCTION void kernel_connect(
      Kokkos::View<cms::kokkos::AtomicPairCounter, KokkosExecSpace> apc1,
      Kokkos::View<cms::kokkos::AtomicPairCounter, KokkosExecSpace> apc2,  // just to zero them,
      TrackingRecHit2DSOAView const *__restrict__ hhp,
      Kokkos::View<GPUCACell *, KokkosExecSpace> cells,
      Kokkos::View<uint32_t, KokkosExecSpace> nCells,
      Kokkos::View<CAConstants::CellNeighborsVector, KokkosExecSpace> cellNeighbors,
      Kokkos::View<GPUCACell::OuterHitOfCell *, KokkosExecSpace> isOuterHitOfCell,
      float hardCurvCut,
      float ptmin,
      float CAThetaCutBarrel,
      float CAThetaCutForward,
      float dcaCutInnerTriplet,
      float dcaCutOuterTriplet,
      const uint32_t stride,
      const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
    const int teamRank = teamMember.team_rank();
    const int teamSize = teamMember.team_size();
    const int leagueRank = teamMember.league_rank();
    const int leagueSize = teamMember.league_size();

    const uint32_t blockDim = teamSize / stride;
    uint32_t first = teamRank % stride;
    uint32_t firstCellIndex = leagueRank * blockDim + teamRank / stride;

    auto const &hh = *hhp;

    if (0 == (firstCellIndex + first)) {
      (apc1()) = 0;
      (apc2()) = 0;
    }  // ready for next kernel

    for (int idx = firstCellIndex, nt = (nCells()); idx < nt; idx += leagueSize * blockDim) {
      auto cellIndex = idx;
      auto &thisCell = cells(idx);
      //if (thisCell.theDoubletId < 0 || thisCell.theUsed>1)
      //  continue;
      auto innerHitId = thisCell.get_inner_hit_id();
      int numberOfPossibleNeighbors = isOuterHitOfCell(innerHitId).size();
      // const and __restrict__ qualifiers to indicate read-only condition
      const auto *__restrict__ vi = isOuterHitOfCell(innerHitId).data();

      constexpr uint32_t last_bpix1_detIndex = 96;
      constexpr uint32_t last_barrel_detIndex = 1184;
      auto ri = thisCell.get_inner_r(hh);
      auto zi = thisCell.get_inner_z(hh);

      auto ro = thisCell.get_outer_r(hh);
      auto zo = thisCell.get_outer_z(hh);
      auto isBarrel = thisCell.get_inner_detIndex(hh) < last_barrel_detIndex;

      for (int j = first; j < numberOfPossibleNeighbors; j += stride) {
        // auto otherCell = __ldg(vi + j); __ldg is a device-only intrinsic. Thus using const and __restrict__
        // qualifiers for vi to increase the likelihood that the compiler will detect the read-only condition.
        auto otherCell = vi[j];
        auto &oc = cells(otherCell);
        // if (cells[otherCell].theDoubletId < 0 ||
        //    cells[otherCell].theUsed>1 )
        //  continue;
        auto r1 = oc.get_inner_r(hh);
        auto z1 = oc.get_inner_z(hh);
        // auto isBarrel = oc.get_outer_detIndex(hh) < last_barrel_detIndex;
        bool aligned = GPUCACell::areAlignedRZ(
            r1,
            z1,
            ri,
            zi,
            ro,
            zo,
            ptmin,
            isBarrel ? CAThetaCutBarrel : CAThetaCutForward);  // 2.f*thetaCut); // FIXME tune cuts
        if (aligned &&
            thisCell.dcaCut(hh,
                            oc,
                            oc.get_inner_detIndex(hh) < last_bpix1_detIndex ? dcaCutInnerTriplet : dcaCutOuterTriplet,
                            hardCurvCut)) {  // FIXME tune cuts
          oc.addOuterNeighbor(cellIndex, cellNeighbors());
          thisCell.theUsed |= 1;
          oc.theUsed |= 1;
        }
      }  // loop on inner cells
    }    // loop on outer cells
  }

  KOKKOS_INLINE_FUNCTION void kernel_find_ntuplets(
      TrackingRecHit2DSOAView const *__restrict__ hhp,
      Kokkos::View<GPUCACell *, KokkosExecSpace> cells,
      Kokkos::View<CAConstants::CellTracksVector, KokkosExecSpace> cellTracks,
      HitContainer *foundNtuplets,
      Kokkos::View<cms::kokkos::AtomicPairCounter, KokkosExecSpace> apc,
      Quality *__restrict__ quality,
      unsigned int minHitsPerNtuplet,
      const size_t idx) {
    // recursive: not obvious to widen
    auto const &hh = *hhp;

    auto const &thisCell = cells(idx);
    if (thisCell.theDoubletId < 0)
      return;  // cut by earlyFishbone

    auto pid = thisCell.theLayerPairId;
    auto doit = minHitsPerNtuplet > 3 ? pid < 3 : pid < 8 || pid > 12;
    if (doit) {
      GPUCACell::TmpTuple stack;
      stack.reset();
      thisCell.find_ntuplets(
          hh, cells.data(), cellTracks(), *foundNtuplets, apc(), quality, stack, minHitsPerNtuplet, pid < 3);
      assert(stack.empty());
      // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
    }
  }

  KOKKOS_INLINE_FUNCTION void kernel_mark_used(TrackingRecHit2DSOAView const *__restrict__ hhp,  // not used
                                               Kokkos::View<GPUCACell *, KokkosExecSpace> cells,
                                               const size_t idx) {
    auto &thisCell = cells(idx);
    if (!thisCell.tracks().empty())
      thisCell.theUsed |= 2;
  }

  KOKKOS_INLINE_FUNCTION void kernel_countMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                                                       Quality const *__restrict__ quality,
                                                       CAConstants::TupleMultiplicity *tupleMultiplicity,
                                                       const int it) {
    auto nhits = foundNtuplets->size(it);
    if (nhits < 3)
      return;
    if (quality[it] == trackQuality::dup)
      return;
    assert(quality[it] == trackQuality::bad);
    if (nhits > 5)
      printf("wrong mult %d %d\n", it, nhits);
    assert(nhits < 8);
    tupleMultiplicity->countDirect(nhits);
  }

  KOKKOS_INLINE_FUNCTION void kernel_fillMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                                                      Quality const *__restrict__ quality,
                                                      CAConstants::TupleMultiplicity *tupleMultiplicity,
                                                      const int it) {
    auto nhits = foundNtuplets->size(it);
    if (nhits < 3)
      return;
    if (quality[it] == trackQuality::dup)
      return;
    assert(quality[it] == trackQuality::bad);
    if (nhits > 5)
      printf("wrong mult %d %d\n", it, nhits);
    assert(nhits < 8);
    tupleMultiplicity->fillDirect(nhits, it);
  }

  KOKKOS_INLINE_FUNCTION void kernel_classifyTracks(HitContainer const *__restrict__ tuples,
                                                    TkSoA const *__restrict__ tracks,
                                                    CAHitNtupletGeneratorKernels::QualityCuts cuts,
                                                    Quality *__restrict__ quality,
                                                    const size_t it) {
    auto nhits = tuples->size(it);
    if (nhits == 0)
      return;  // guard

    // if duplicate: not even fit
    if (quality[it] == trackQuality::dup)
      return;

    assert(quality[it] == trackQuality::bad);

    // mark doublets as bad
    if (nhits < 3)
      return;

    // if the fit has any invalid parameters, mark it as bad
    bool isNaN = false;
    for (int i = 0; i < 5; ++i) {
      isNaN |= std::isnan(tracks->stateAtBS.state(it)(i));
    }
    if (isNaN) {
#ifdef NTUPLE_DEBUG
      printf("NaN in fit %d size %d chi2 %f\n", it, tuples->size(it), tracks->chi2(it));
#endif
      return;
    }

    // compute a pT-dependent chi2 cut
    // default parameters:
    //   - chi2MaxPt = 10 GeV
    //   - chi2Coeff = { 0.68177776, 0.74609577, -0.08035491, 0.00315399 }
    //   - chi2Scale = 30 for broken line fit, 45 for Riemann fit
    // (see CAHitNtupletGeneratorGPU.cc)
    float pt = std::min<float>(tracks->pt(it), cuts.chi2MaxPt);
    float chi2Cut = cuts.chi2Scale *
                    (cuts.chi2Coeff[0] + pt * (cuts.chi2Coeff[1] + pt * (cuts.chi2Coeff[2] + pt * cuts.chi2Coeff[3])));
    // above number were for Quads not normalized so for the time being just multiple by ndof for Quads  (triplets to be understood)
    if (3.f * tracks->chi2(it) >= chi2Cut) {
#ifdef NTUPLE_DEBUG
      printf("Bad fit %d size %d pt %f eta %f chi2 %f\n",
             it,
             tuples->size(it),
             tracks->pt(it),
             tracks->eta(it),
             3.f * tracks->chi2(it));
#endif
      return;
    }

    // impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
    // default cuts:
    //   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
    //   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
    // (see CAHitNtupletGeneratorGPU.cc)
    auto const &region = (nhits > 3) ? cuts.quadruplet : cuts.triplet;
    bool isOk = (std::abs(tracks->tip(it)) < region.maxTip) and (tracks->pt(it) > region.minPt) and
                (std::abs(tracks->zip(it)) < region.maxZip);

    if (isOk)
      quality[it] = trackQuality::loose;
  }

  KOKKOS_INLINE_FUNCTION void kernel_doStatsForTracks(HitContainer const *__restrict__ tuples,
                                                      Quality const *__restrict__ quality,
                                                      CAHitNtupletGeneratorKernels::Counters *counters,
                                                      const size_t idx) {
    if (tuples->size(idx) == 0)
      return;  //guard
    if (quality[idx] != trackQuality::loose)
      return;
    Kokkos::atomic_add<unsigned long long>(&(counters->nGoodTracks), 1);
  }

  KOKKOS_INLINE_FUNCTION void kernel_countHitInTracks(HitContainer const *__restrict__ tuples,
                                                      Quality const *__restrict__ quality,
                                                      CAHitNtupletGeneratorKernels::HitToTuple *hitToTuple,
                                                      const size_t idx) {
    if (tuples->size(idx) == 0)
      return;  // guard
    if (quality[idx] != trackQuality::loose)
      return;
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->countDirect(*h);
  }

  KOKKOS_INLINE_FUNCTION void kernel_fillHitInTracks(HitContainer const *__restrict__ tuples,
                                                     Quality const *__restrict__ quality,
                                                     CAHitNtupletGeneratorKernels::HitToTuple *hitToTuple,
                                                     const size_t idx) {
    if (tuples->size(idx) == 0)
      return;  // guard
    if (quality[idx] != trackQuality::loose)
      return;
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->fillDirect(*h, idx);
  }

  KOKKOS_INLINE_FUNCTION void kernel_fillHitDetIndices(HitContainer const *__restrict__ tuples,
                                                       TrackingRecHit2DSOAView const *__restrict__ hhp,
                                                       HitContainer *__restrict__ hitDetIndices,
                                                       const size_t idx) {
    // this could be more efficienctly implemented with thread teams, wrt. CUDA implementation
    if (idx < tuples->totbins()) {
      hitDetIndices->off[idx] = tuples->off[idx];
    }
    if (idx < tuples->size()) {
      auto const &hh = *hhp;
      assert(tuples->bins[idx] < hh.nHits());
      hitDetIndices->bins[idx] = hh.detectorIndex(tuples->bins[idx]);
    }
  }

  KOKKOS_INLINE_FUNCTION void kernel_doStatsForHitInTracks(
      CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ hitToTuple,
      CAHitNtupletGeneratorKernels::Counters *counters,
      const size_t idx) {
    auto &c = *counters;
    if (hitToTuple->size(idx) == 0)
      return;
    Kokkos::atomic_add<unsigned long long>(&c.nUsedHits, 1);
    if (hitToTuple->size(idx) > 1)
      Kokkos::atomic_add<unsigned long long>(&c.nDupHits, 1);
  }

  KOKKOS_INLINE_FUNCTION void kernel_tripletCleaner(
      TrackingRecHit2DSOAView const *__restrict__ hhp,
      HitContainer const *__restrict__ ptuples,
      TkSoA const *__restrict__ ptracks,
      Quality *__restrict__ quality,
      CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ phitToTuple,
      const size_t idx) {
    constexpr auto bad = trackQuality::bad;
    constexpr auto dup = trackQuality::dup;
    // constexpr auto loose = trackQuality::loose;

    auto &hitToTuple = *phitToTuple;
    auto const &foundNtuplets = *ptuples;
    auto const &tracks = *ptracks;

    if (hitToTuple.size(idx) < 2)
      return;

    float mc = 10000.f;
    uint16_t im = 60000;
    uint32_t maxNh = 0;

    // find maxNh
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      uint32_t nh = foundNtuplets.size(*it);
      maxNh = std::max(nh, maxNh);
    }
    // kill all tracks shorter than maxHn (only triplets???)
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      uint32_t nh = foundNtuplets.size(*it);
      if (maxNh != nh)
        quality[*it] = dup;
    }

    if (maxNh > 3)
      return;
    // if (idx>=l1end) continue;  // only for layer 1
    // for triplets choose best tip!
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] != bad && std::abs(tracks.tip(it)) < mc) {
        mc = std::abs(tracks.tip(it));
        im = it;
      }
    }
    // mark duplicates
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] != bad && it != im)
        quality[it] = dup;  //no race:  simple assignment of the same constant
    }
  }

  KOKKOS_INLINE_FUNCTION void kernel_print_found_ntuplets(
      TrackingRecHit2DSOAView const *__restrict__ hhp,
      HitContainer const *__restrict__ ptuples,
      TkSoA const *__restrict__ ptracks,
      Quality const *__restrict__ quality,
      CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ phitToTuple,
      uint32_t maxPrint,
      int iev,
      const size_t i) {
    auto const &foundNtuplets = *ptuples;
    auto const &tracks = *ptracks;
    auto nh = foundNtuplets.size(i);
    if (nh < 3)
      return;
    printf("TK: %d %d %d %f %f %f %f %f %f %f %d %d %d %d %d\n",
           int(10000 * iev + i),
           int(quality[i]),
           nh,
           tracks.charge(i),
           tracks.pt(i),
           tracks.eta(i),
           tracks.phi(i),
           tracks.tip(i),
           tracks.zip(i),
           //           asinhf(fit_results[i].par(3)),
           tracks.chi2(i),
           *foundNtuplets.begin(i),
           *(foundNtuplets.begin(i) + 1),
           *(foundNtuplets.begin(i) + 2),
           nh > 3 ? int(*(foundNtuplets.begin(i) + 3)) : -1,
           nh > 4 ? int(*(foundNtuplets.begin(i) + 4)) : -1);
  }
#ifdef TODO
  __global__ void kernel_printCounters(cAHitNtupletGenerator::Counters const *counters) {
    auto const &c = *counters;
    printf(
        "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nGoodTracks | nUsedHits | nDupHits | "
        "nKilledCells | "
        "nEmptyCells | nZeroTrackCells ||\n");
    printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
           c.nEvents,
           c.nHits,
           c.nCells,
           c.nTuples,
           c.nGoodTracks,
           c.nFitTracks,
           c.nUsedHits,
           c.nDupHits,
           c.nKilledCells,
           c.nEmptyCells,
           c.nZeroTrackCells);
    printf("Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f||\n",
           c.nEvents,
           c.nHits / double(c.nEvents),
           c.nCells / double(c.nEvents),
           c.nTuples / double(c.nEvents),
           c.nFitTracks / double(c.nEvents),
           c.nGoodTracks / double(c.nEvents),
           c.nUsedHits / double(c.nEvents),
           c.nDupHits / double(c.nEvents),
           c.nKilledCells / double(c.nEvents),
           c.nEmptyCells / double(c.nCells),
           c.nZeroTrackCells / double(c.nCells));
  }
#endif
}  // namespace KOKKOS_NAMESPACE
