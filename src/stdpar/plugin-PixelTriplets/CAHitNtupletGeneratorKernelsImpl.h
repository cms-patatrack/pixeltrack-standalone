//
// Original Author: Felice Pantaleo, CERN
//

// #define NTUPLE_DEBUG

#include <algorithm>
#include <atomic>
#include <ranges>
#include <execution>
#include <cmath>
#include <cstdint>

#include "CondFormats/pixelCPEforGPU.h"

#include "CAConstants.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

using HitsOnGPU = TrackingRecHit2DSOAView;
using HitsOnCPU = TrackingRecHit2D;

using HitToTuple = CAConstants::HitToTuple;
using TupleMultiplicity = CAConstants::TupleMultiplicity;

using Quality = pixelTrack::Quality;
using TkSoA = pixelTrack::TrackSoA;
using HitContainer = pixelTrack::HitContainer;

void kernel_checkOverflows(HitContainer const *foundNtuplets,
                                      CAConstants::TupleMultiplicity *tupleMultiplicity,
                                      cms::cuda::AtomicPairCounter *apc,
                                      GPUCACell const *__restrict__ cells,
                                      uint32_t const *__restrict__ nCells,
                                      gpuPixelDoublets::CellNeighborsVector const *cellNeighbors,
                                      gpuPixelDoublets::CellTracksVector const *cellTracks,
                                      GPUCACell::OuterHitOfCell const *__restrict__ isOuterHitOfCell,
                                      uint32_t nHits,
                                      uint32_t maxNumberOfDoublets,
                                      CAHitNtupletGeneratorKernelsGPU::Counters *counters) {

  auto &c = *counters;
  // counters once per event
  std::atomic_ref arNEvents {c.nEvents};
  arNEvents++;
  std::atomic_ref arNHits {c.nHits};
  arNHits += nHits;
  std::atomic_ref arNCells {c.nCells};
  arNCells += *nCells;
  std::atomic_ref arNTuples {c.nTuples};
  arNTuples += apc->get().m;
  std::atomic_ref arNFitTtracks {c.nFitTracks};
  arNFitTtracks += tupleMultiplicity->size();

#ifdef NTUPLE_DEBUG
  printf("number of found cells %d, found tuples %d with total hits %d out of %d\n",
         *nCells,
         apc->get().m,
         apc->get().n,
         nHits);
  if (apc->get().m < CAConstants::maxNumberOfQuadruplets()) {
    assert(foundNtuplets->size(apc->get().m) == 0);
    assert(foundNtuplets->size() == apc->get().n);
  }

  for (int idx = 0, nt = foundNtuplets->nbins(); idx < nt; ++idx) {
    if (foundNtuplets->size(idx) > 5)
      printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
    assert(foundNtuplets->size(idx) < 6);
    for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
      assert(*ih < nHits);
  }
#endif

  if (apc->get().m >= CAConstants::maxNumberOfQuadruplets())
    printf("Tuples overflow\n");
  if (*nCells >= maxNumberOfDoublets)
    printf("Cells overflow\n");
  if (cellNeighbors && cellNeighbors->full())
    printf("cellNeighbors overflow\n");
  if (cellTracks && cellTracks->full())
    printf("cellTracks overflow\n");

  auto iter{std::views::iota(0U, *nCells)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto const &thisCell = cells[idx];
    if (thisCell.outerNeighbors().full())  //++tooManyNeighbors[thisCell.theLayerPairId];
      printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
    if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
      printf("Tracks overflow %d in %d\n", idx, thisCell.theLayerPairId);
    if (thisCell.theDoubletId < 0) {
      std::atomic_ref arNKilledCells{counters->nKilledCells};
      arNKilledCells++;
    }
    if (0 == thisCell.theUsed) {
      std::atomic_ref arNEmptyCells{counters->nEmptyCells};
      arNEmptyCells++;
    }
    if (thisCell.tracks().empty()) {
      std::atomic_ref arNZeroTrackCells{counters->nZeroTrackCells};
      arNZeroTrackCells++;
    }
  });

  auto iter_nh{std::views::iota(0U, nHits)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    if (isOuterHitOfCell[idx].full())  // ++tooManyOuterHitOfCell;
      printf("OuterHitOfCell overflow %d\n", idx);
  });
}

void kernel_fishboneCleaner(GPUCACell const *cells, uint32_t const *__restrict__ nCells, Quality *quality) {
  constexpr auto bad = trackQuality::bad;

  auto iter{std::views::iota(0U, *nCells)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto const &thisCell = cells[idx];
    if (thisCell.theDoubletId >= 0)
      return;

    for (auto it : thisCell.tracks())
      quality[it] = bad;
  });
}

void kernel_earlyDuplicateRemover(GPUCACell const *cells,
                                             uint32_t const *__restrict__ nCells,
                                             HitContainer *foundNtuplets,
                                             Quality *quality) {
  // constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  // constexpr auto loose = trackQuality::loose;

  assert(nCells);
  auto iter{std::views::iota(0U, *nCells)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto const &thisCell = cells[idx];

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
  });
}

void kernel_fastDuplicateRemover(GPUCACell const *__restrict__ cells,
                                            uint32_t const *__restrict__ nCells,
                                            HitContainer const *__restrict__ foundNtuplets,
                                            TkSoA *__restrict__ tracks) {
  constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  constexpr auto loose = trackQuality::loose;

  assert(nCells);

  auto iter{std::views::iota(0U, *nCells)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto const &thisCell = cells[idx];
    if (thisCell.tracks().size() < 2)
      return;
    // if (thisCell.theDoubletId < 0) continue;

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
  });
}

void kernel_connect(cms::cuda::AtomicPairCounter *apc1,
                    cms::cuda::AtomicPairCounter *apc2,  // just to zero them,
                    GPUCACell::Hits const *__restrict__ hhp,
                    GPUCACell *cells,
                    uint32_t const *__restrict__ nCells,
                    gpuPixelDoublets::CellNeighborsVector *cellNeighbors,
                    GPUCACell::OuterHitOfCell const *__restrict__ isOuterHitOfCell,
                    float hardCurvCut,
                    float ptmin,
                    float CAThetaCutBarrel,
                    float CAThetaCutForward,
                    float dcaCutInnerTriplet,
                    float dcaCutOuterTriplet) {

  (*apc1) = 0;
  (*apc2) = 0;
  // ready for next kernel
  auto iter{std::views::iota(0U, *nCells)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto const &hh = *hhp;
    auto cellIndex = idx;
    auto &thisCell = cells[idx];
    //if (thisCell.theDoubletId < 0 || thisCell.theUsed>1)
    //  continue;
    auto innerHitId = thisCell.get_inner_hit_id();
    int numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
    auto vi = isOuterHitOfCell[innerHitId].data();

    constexpr uint32_t last_bpix1_detIndex = 96;
    constexpr uint32_t last_barrel_detIndex = 1184;
    auto ri = thisCell.get_inner_r(hh);
    auto zi = thisCell.get_inner_z(hh);

    auto ro = thisCell.get_outer_r(hh);
    auto zo = thisCell.get_outer_z(hh);
    auto isBarrel = thisCell.get_inner_detIndex(hh) < last_barrel_detIndex;

    for (int j = 0; j < numberOfPossibleNeighbors; ++j) {
      auto otherCell = *(vi + j);
      auto &oc = cells[otherCell];
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
        oc.addOuterNeighbor(cellIndex, *cellNeighbors);
        thisCell.theUsed |= 1;
        oc.theUsed |= 1;
      }
    }  // loop on inner cells
  });    // loop on outer cells
}

void kernel_find_ntuplets(GPUCACell::Hits const *__restrict__ hhp,
                                     GPUCACell *__restrict__ cells,
                                     uint32_t const *nCells,
                                     gpuPixelDoublets::CellTracksVector *cellTracks,
                                     HitContainer *foundNtuplets,
                                     cms::cuda::AtomicPairCounter *apc,
                                     Quality *__restrict__ quality,
                                     unsigned int minHitsPerNtuplet) {
  // recursive: not obvious to widen

  auto iter{std::views::iota(0U, *nCells)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto const &hh = *hhp;
    auto const &thisCell = cells[idx];
    if (thisCell.theDoubletId < 0)
      return;  // cut by earlyFishbone

    auto pid = thisCell.theLayerPairId;
    auto doit = minHitsPerNtuplet > 3 ? pid < 3 : pid < 8 || pid > 12;
    if (doit) {
      GPUCACell::TmpTuple stack;
      stack.reset();
      thisCell.find_ntuplets<6>(hh, cells, *cellTracks, *foundNtuplets, *apc, quality, stack, minHitsPerNtuplet, pid < 3);
      assert(stack.empty());
      // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
    }
  });
}

void kernel_mark_used(GPUCACell::Hits const *__restrict__ hhp,
                                 GPUCACell *__restrict__ cells,
                                 uint32_t const *nCells) {
  // auto const &hh = *hhp;
  auto iter{std::views::iota(0U, *nCells)};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto &thisCell = cells[idx];
    if (!thisCell.tracks().empty())
      thisCell.theUsed |= 2;
  });
}

void kernel_countMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                                         Quality const *__restrict__ quality,
                                         CAConstants::TupleMultiplicity *tupleMultiplicity) {
  auto iter{std::views::iota(0U, foundNtuplets->nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto it) {
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
  });
}

void kernel_fillMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                                        Quality const *__restrict__ quality,
                                        CAConstants::TupleMultiplicity *tupleMultiplicity) {
  auto iter{std::views::iota(0U, foundNtuplets->nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto it) {
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
  });
}

void kernel_classifyTracks(HitContainer const *__restrict__ tuples,
                                      TkSoA const *__restrict__ tracks,
                                      CAHitNtupletGeneratorKernelsGPU::QualityCuts cuts,
                                      Quality *__restrict__ quality) {
  auto iter{std::views::iota(0U, tuples->nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto it) {
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
  });
}

void kernel_doStatsForTracks(HitContainer const *__restrict__ tuples,
                                        Quality const *__restrict__ quality,
                                        CAHitNtupletGeneratorKernelsGPU::Counters *counters) {
  auto iter{std::views::iota(0U, tuples->nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    if (tuples->size(idx) == 0 || quality[idx] != trackQuality::loose) {
      return;
    }
    std::atomic_ref ar{counters->nGoodTracks};
    ++ar;
  });
}

void kernel_countHitInTracks(HitContainer const *__restrict__ tuples,
                                        Quality const *__restrict__ quality,
                                        CAHitNtupletGeneratorKernelsGPU::HitToTuple *hitToTuple) {
  auto iter{std::views::iota(0U, tuples->nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    if (tuples->size(idx) == 0 || quality[idx] != trackQuality::loose)
      return;  // guard
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->countDirect(*h);
  });
}

void kernel_fillHitInTracks(HitContainer const *__restrict__ tuples,
                                       Quality const *__restrict__ quality,
                                       CAHitNtupletGeneratorKernelsGPU::HitToTuple *hitToTuple) {
  auto iter{std::views::iota(0U, tuples->nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    if (tuples->size(idx) == 0 || quality[idx] != trackQuality::loose)
      return;  // guard
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->fillDirect(*h, idx);
  });
}

void kernel_fillHitDetIndices(HitContainer const *__restrict__ tuples,
                                         TrackingRecHit2DSOAView const *__restrict__ hhp,
                                         HitContainer *__restrict__ hitDetIndices) {
  // copy offsets
  std::copy(std::execution::par, tuples->off, tuples->off + tuples->totbins(), hitDetIndices->off);
  // fill hit indices
  auto iter{std::views::iota(0U, tuples->size())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto const &hh = *hhp;
    assert(tuples->bins[idx] < hh.nHits());
    hitDetIndices->bins[idx] = hh.detectorIndex(tuples->bins[idx]);
  });
}

void kernel_doStatsForHitInTracks(CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ hitToTuple,
                                             CAHitNtupletGeneratorKernelsGPU::Counters *counters) {
  auto iter{std::views::iota(0U, hitToTuple->nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto &c = *counters;
    if (hitToTuple->size(idx) == 0)
      return;
    std::atomic_ref ar_nUsed{c.nUsedHits};
    ++ar_nUsed;
    if (hitToTuple->size(idx) > 1){
      std::atomic_ref ar_nDup{c.nDupHits};
      ++ar_nDup;
    }
  });
}

void kernel_tripletCleaner(TrackingRecHit2DSOAView const *__restrict__ hhp,
                                      HitContainer const *__restrict__ ptuples,
                                      TkSoA const *__restrict__ ptracks,
                                      Quality *__restrict__ quality,
                                      CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple) {
  constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  // constexpr auto loose = trackQuality::loose;


  //  auto const & hh = *hhp;
  // auto l1end = hh.hitsLayerStart_d[1];
  auto iter{std::views::iota(0U, phitToTuple->nbins())};
  std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto idx) {
    auto &hitToTuple = *phitToTuple;
    auto const &foundNtuplets = *ptuples;
    auto const &tracks = *ptracks;
  //for (int idx = 0, ntot = hitToTuple.nbins(); idx < ntot; idx += gridDim.x * blockDim.x) {
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
  });  // loop over hits
}

void kernel_print_found_ntuplets(TrackingRecHit2DSOAView const *__restrict__ hhp,
                                            HitContainer const *__restrict__ ptuples,
                                            TkSoA const *__restrict__ ptracks,
                                            Quality const *__restrict__ quality,
                                            CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple,
                                            uint32_t maxPrint,
                                            int iev) {
  auto const &foundNtuplets = *ptuples;
  auto const &tracks = *ptracks;
  for (int i = 0, np = std::min(maxPrint, foundNtuplets.nbins()); i < np; ++i) {
    auto nh = foundNtuplets.size(i);
    if (nh < 3)
      continue;
    printf("TK: %d %d %d %f %f %f %f %f %f %f %d %d %d %d %d\n",
           10000 * iev + i,
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
}

void kernel_printCounters(cAHitNtupletGenerator::Counters const *counters) {
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
