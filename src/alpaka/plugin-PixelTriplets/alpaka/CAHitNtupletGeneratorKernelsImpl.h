//
// Original Author: Felice Pantaleo, CERN
//

// #define NTUPLE_DEBUG

#include <cmath>
#include <cstdint>

#include "AlpakaCore/alpakaKernelCommon.h"

#include "CondFormats/pixelCPEforGPU.h"

#include "CAConstants.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DAlpaka;

  using HitToTuple = CAConstants::HitToTuple;
  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

  struct kernel_checkOverflows {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  HitContainer const *foundNtuplets,
                                  CAConstants::TupleMultiplicity *tupleMultiplicity,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  GPUCACell const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  gpuPixelDoublets::CellNeighborsVector const *cellNeighbors,
                                  gpuPixelDoublets::CellTracksVector const *cellTracks,
                                  GPUCACell::OuterHitOfCell const *__restrict__ isOuterHitOfCell,
                                  uint32_t nHits,
                                  uint32_t maxNumberOfDoublets,
                                  CAHitNtupletGeneratorKernels::Counters *counters) const {
      const uint32_t threadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

      auto &c = *counters;
      // counters once per event
      if (0 == threadIdx) {
        alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nEvents, 1ull);
        alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nHits, static_cast<unsigned long long>(nHits));
        alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nCells, static_cast<unsigned long long>(*nCells));
        alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nTuples, static_cast<unsigned long long>(apc->get().m));
        alpaka::atomicOp<alpaka::AtomicAdd>(
            acc, &c.nFitTracks, static_cast<unsigned long long>(tupleMultiplicity->size()));
      }

#ifdef NTUPLE_DEBUG
      if (0 == threadIdx) {
        printf("number of found cells %d, found tuples %d with total hits %d out of %d\n",
               *nCells,
               apc->get().m,
               apc->get().n,
               nHits);
        if (apc->get().m < CAConstants::maxNumberOfQuadruplets()) {
          assert(foundNtuplets->size(apc->get().m) == 0);
          assert(foundNtuplets->size() == apc->get().n);
        }
      }

      const auto ntNbins = foundNtuplets->nbins();
      cms::alpakatools::for_each_element_in_grid_strided(acc, ntNbins, [&](uint32_t idx) {
        if (foundNtuplets->size(idx) > 5)
          printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
        assert(foundNtuplets->size(idx) < 6);
        for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
          assert(*ih < nHits);
      });
#endif

      if (0 == threadIdx) {
        if (apc->get().m >= CAConstants::maxNumberOfQuadruplets())
          printf("Tuples overflow\n");
        if (*nCells >= maxNumberOfDoublets)
          printf("Cells overflow\n");
        if (cellNeighbors && cellNeighbors->full())
          printf("cellNeighbors overflow\n");
        if (cellTracks && cellTracks->full())
          printf("cellTracks overflow\n");
      }

      const auto ntNCells = (*nCells);
      cms::alpakatools::for_each_element_in_grid_strided(acc, ntNCells, [&](uint32_t idx) {
        auto const &thisCell = cells[idx];
        if (thisCell.outerNeighbors().full())  //++tooManyNeighbors[thisCell.theLayerPairId];
          printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
        if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
          printf("Tracks overflow %d in %d\n", idx, thisCell.theLayerPairId);
        if (thisCell.theDoubletId < 0)
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nKilledCells, 1ull);
        if (0 == thisCell.theUsed)
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nEmptyCells, 1ull);
        if (thisCell.tracks().empty())
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nZeroTrackCells, 1ull);
      });

      cms::alpakatools::for_each_element_in_grid_strided(acc, nHits, [&](uint32_t idx) {
        if (isOuterHitOfCell[idx].full())  // ++tooManyOuterHitOfCell;
          printf("OuterHitOfCell overflow %d\n", idx);
      });
    }
  };

  struct kernel_fishboneCleaner {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  GPUCACell const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  Quality *quality) const {
      constexpr auto bad = trackQuality::bad;

      const auto ntNCells = (*nCells);
      cms::alpakatools::for_each_element_in_grid_strided(acc, ntNCells, [&](uint32_t idx) {
        auto const &thisCell = cells[idx];

        if (thisCell.theDoubletId < 0) {
          for (auto it : thisCell.tracks())
            quality[it] = bad;
        }
      });
    }
  };

  struct kernel_earlyDuplicateRemover {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  GPUCACell const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  HitContainer *foundNtuplets,
                                  Quality *quality) const {
      // constexpr auto bad = trackQuality::bad;
      constexpr auto dup = trackQuality::dup;
      // constexpr auto loose = trackQuality::loose;

      assert(nCells);
      const auto ntNCells = (*nCells);
      cms::alpakatools::for_each_element_in_grid_strided(acc, ntNCells, [&](uint32_t idx) {
        auto const &thisCell = cells[idx];

        if (thisCell.tracks().size() >= 2) {
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
      });
    }
  };

  struct kernel_fastDuplicateRemover {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  GPUCACell const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TkSoA *__restrict__ tracks) const {
      constexpr auto bad = trackQuality::bad;
      constexpr auto dup = trackQuality::dup;
      constexpr auto loose = trackQuality::loose;

      assert(nCells);

      cms::alpakatools::for_each_element_in_grid_strided(acc, (*nCells), [&](uint32_t idx) {
        auto const &thisCell = cells[idx];
        if (thisCell.tracks().size() >= 2) {
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
        }
      });
    }
  };

  struct kernel_connect {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  cms::alpakatools::AtomicPairCounter *apc1,
                                  cms::alpakatools::AtomicPairCounter *apc2,  // just to zero them,
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
                                  float dcaCutOuterTriplet) const {
      auto const &hh = *hhp;

      const uint32_t dimIndexY = 0u;
      const uint32_t dimIndexX = 1u;
      const uint32_t threadIdxY(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[dimIndexY]);
      const uint32_t threadIdxLocalX(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndexX]);

      if (0 == (threadIdxY + threadIdxLocalX)) {
        (*apc1) = 0;
        (*apc2) = 0;
      }  // ready for next kernel

      cms::alpakatools::for_each_element_in_grid_strided(
          acc,
          (*nCells),
          0u,
          [&](uint32_t idx) {
            auto cellIndex = idx;
            auto &thisCell = cells[idx];
            //if (thisCell.theDoubletId < 0 || thisCell.theUsed>1)
            //  continue;
            auto innerHitId = thisCell.get_inner_hit_id();
            int numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
            const auto *__restrict__ vi = isOuterHitOfCell[innerHitId].data();

            constexpr uint32_t last_bpix1_detIndex = 96;
            constexpr uint32_t last_barrel_detIndex = 1184;
            auto ri = thisCell.get_inner_r(hh);
            auto zi = thisCell.get_inner_z(hh);

            auto ro = thisCell.get_outer_r(hh);
            auto zo = thisCell.get_outer_z(hh);
            auto isBarrel = thisCell.get_inner_detIndex(hh) < last_barrel_detIndex;

            cms::alpakatools::for_each_element_in_block_strided(
                acc,
                numberOfPossibleNeighbors,
                0u,
                [&](uint32_t j) {
                  auto otherCell = vi[j];  // NB: Was with __ldg in legacy
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
                  if (aligned && thisCell.dcaCut(hh,
                                                 oc,
                                                 oc.get_inner_detIndex(hh) < last_bpix1_detIndex ? dcaCutInnerTriplet
                                                                                                 : dcaCutOuterTriplet,
                                                 hardCurvCut)) {  // FIXME tune cuts
                    oc.addOuterNeighbor(acc, cellIndex, *cellNeighbors);
                    thisCell.theUsed |= 1;
                    oc.theUsed |= 1;
                  }
                },
                dimIndexX);  // loop on inner cells
          },
          dimIndexY);  // loop on outer cells
    }
  };

  struct kernel_find_ntuplets {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  GPUCACell::Hits const *__restrict__ hhp,
                                  GPUCACell *__restrict__ cells,
                                  uint32_t const *nCells,
                                  gpuPixelDoublets::CellTracksVector *cellTracks,
                                  HitContainer *foundNtuplets,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  Quality *__restrict__ quality,
                                  unsigned int minHitsPerNtuplet) const {
      // recursive: not obvious to widen
      auto const &hh = *hhp;

      //auto first = threadIdx.x + blockIdx.x * blockDim.x;
      //for (int idx = first, nt = (*nCells); idx < nt; idx += gridDim.x * blockDim.x) {
      cms::alpakatools::for_each_element_in_grid_strided(acc, (*nCells), [&](uint32_t idx) {
        auto const &thisCell = cells[idx];
        if (thisCell.theDoubletId >= 0) {  // cut by earlyFishbone

          auto pid = thisCell.theLayerPairId;
          auto doit = minHitsPerNtuplet > 3 ? pid < 3 : pid < 8 || pid > 12;
          if (doit) {
            GPUCACell::TmpTuple stack;
            stack.reset();
            thisCell.find_ntuplets(
                acc, hh, cells, *cellTracks, *foundNtuplets, *apc, quality, stack, minHitsPerNtuplet, pid < 3);
            assert(stack.empty());
            // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
          }
        }
      });
    }
  };

  struct kernel_mark_used {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  GPUCACell::Hits const *__restrict__ hhp,
                                  GPUCACell *__restrict__ cells,
                                  uint32_t const *nCells) const {
      // auto const &hh = *hhp;
      cms::alpakatools::for_each_element_in_grid_strided(acc, (*nCells), [&](uint32_t idx) {
        auto &thisCell = cells[idx];
        if (!thisCell.tracks().empty())
          thisCell.theUsed |= 2;
      });
    }
  };

  struct kernel_countMultiplicity {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  Quality const *__restrict__ quality,
                                  CAConstants::TupleMultiplicity *tupleMultiplicity) const {
      cms::alpakatools::for_each_element_in_grid_strided(acc, foundNtuplets->nbins(), [&](uint32_t it) {
        auto nhits = foundNtuplets->size(it);
        if (nhits >= 3 && quality[it] != trackQuality::dup) {
          assert(quality[it] == trackQuality::bad);
          if (nhits > 5)
            printf("wrong mult %d %d\n", it, nhits);
          assert(nhits < 8);
          tupleMultiplicity->countDirect(acc, nhits);
        }
      });
    }
  };

  struct kernel_fillMultiplicity {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  Quality const *__restrict__ quality,
                                  CAConstants::TupleMultiplicity *tupleMultiplicity) const {
      cms::alpakatools::for_each_element_in_grid_strided(acc, foundNtuplets->nbins(), [&](uint32_t it) {
        auto nhits = foundNtuplets->size(it);
        if (nhits >= 3 && quality[it] != trackQuality::dup) {
          assert(quality[it] == trackQuality::bad);
          if (nhits > 5)
            printf("wrong mult %d %d\n", it, nhits);
          assert(nhits < 8);
          tupleMultiplicity->fillDirect(acc, nhits, it);
        }
      });
    }
  };

  struct kernel_classifyTracks {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  HitContainer const *__restrict__ tuples,
                                  TkSoA const *__restrict__ tracks,
                                  CAHitNtupletGeneratorKernels::QualityCuts cuts,
                                  Quality *__restrict__ quality) const {
      cms::alpakatools::for_each_element_in_grid_strided(acc, tuples->nbins(), [&](uint32_t it) {
        auto nhits = tuples->size(it);
        if (nhits == 0)
          return;  // guard

        // if duplicate: not even fit
        // mark doublets as bad
        if (quality[it] != trackQuality::dup && nhits >= 3) {
          assert(quality[it] == trackQuality::bad);

          // if the fit has any invalid parameters, mark it as bad
          bool isNaN = false;
          for (int i = 0; i < 5; ++i) {
            isNaN |= std::isnan(tracks->stateAtBS.state(it)(i));
          }
          if (!isNaN) {
#ifdef NTUPLE_DEBUG
            printf("NaN in fit %d size %d chi2 %f\n", it, tuples->size(it), tracks->chi2(it));
#endif

            // compute a pT-dependent chi2 cut
            // default parameters:
            //   - chi2MaxPt = 10 GeV
            //   - chi2Coeff = { 0.68177776, 0.74609577, -0.08035491, 0.00315399 }
            //   - chi2Scale = 30 for broken line fit, 45 for Riemann fit
            // (see CAHitNtupletGeneratorGPU.cc)
            float pt = std::min<float>(tracks->pt(it), cuts.chi2MaxPt);
            float chi2Cut =
                cuts.chi2Scale *
                (cuts.chi2Coeff[0] + pt * (cuts.chi2Coeff[1] + pt * (cuts.chi2Coeff[2] + pt * cuts.chi2Coeff[3])));
            // above number were for Quads not normalized so for the time being just multiple by ndof for Quads  (triplets to be understood)
            if (3.f * tracks->chi2(it) < chi2Cut) {
#ifdef NTUPLE_DEBUG
              printf("Bad fit %d size %d pt %f eta %f chi2 %f\n",
                     it,
                     tuples->size(it),
                     tracks->pt(it),
                     tracks->eta(it),
                     3.f * tracks->chi2(it));
#endif

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

            }  // chi2Cut
          }    // !isNaN
        }      // trackQuality and nhits
      });
    }
  };

  struct kernel_doStatsForTracks {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  HitContainer const *__restrict__ tuples,
                                  Quality const *__restrict__ quality,
                                  CAHitNtupletGeneratorKernels::Counters *counters) const {
      cms::alpakatools::for_each_element_in_grid_strided(acc, tuples->nbins(), [&](uint32_t idx) {
        if (tuples->size(idx) == 0)
          return;  //guard
        if (quality[idx] == trackQuality::loose) {
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, &(counters->nGoodTracks), 1ull);
        }
      });
    }
  };

  struct kernel_countHitInTracks {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  HitContainer const *__restrict__ tuples,
                                  Quality const *__restrict__ quality,
                                  CAHitNtupletGeneratorKernels::HitToTuple *hitToTuple) const {
      cms::alpakatools::for_each_element_in_grid_strided(acc, tuples->nbins(), [&](uint32_t idx) {
        if (tuples->size(idx) == 0)
          return;  // guard
        if (quality[idx] == trackQuality::loose) {
          for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
            hitToTuple->countDirect(acc, *h);
        }
      });
    }
  };

  struct kernel_fillHitInTracks {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  HitContainer const *__restrict__ tuples,
                                  Quality const *__restrict__ quality,
                                  CAHitNtupletGeneratorKernels::HitToTuple *hitToTuple) const {
      cms::alpakatools::for_each_element_in_grid_strided(acc, tuples->nbins(), [&](uint32_t idx) {
        if (tuples->size(idx) == 0)
          return;  // guard
        if (quality[idx] == trackQuality::loose) {
          for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
            hitToTuple->fillDirect(acc, *h, idx);
        }
      });
    }
  };

  struct kernel_fillHitDetIndices {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  HitContainer const *__restrict__ tuples,
                                  TrackingRecHit2DSOAView const *__restrict__ hhp,
                                  HitContainer *__restrict__ hitDetIndices) const {
      // copy offsets
      cms::alpakatools::for_each_element_in_grid_strided(
          acc, tuples->totbins(), [&](uint32_t idx) { hitDetIndices->off[idx] = tuples->off[idx]; });
      // fill hit indices
      auto const &hh = *hhp;
#ifndef NDEBUG
      auto nhits = hh.nHits();
#endif
      cms::alpakatools::for_each_element_in_grid_strided(acc, tuples->size(), [&](uint32_t idx) {
        assert(tuples->bins[idx] < nhits);
        hitDetIndices->bins[idx] = hh.detectorIndex(tuples->bins[idx]);
      });
    }
  };

  struct kernel_doStatsForHitInTracks {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ hitToTuple,
                                  CAHitNtupletGeneratorKernels::Counters *counters) const {
      auto &c = *counters;
      cms::alpakatools::for_each_element_in_grid_strided(acc, hitToTuple->nbins(), [&](uint32_t idx) {
        if (hitToTuple->size(idx) != 0) {  // SHALL NOT BE break
          alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nUsedHits, 1ull);
          if (hitToTuple->size(idx) > 1)
            alpaka::atomicOp<alpaka::AtomicAdd>(acc, &c.nDupHits, 1ull);
        }
      });
    }
  };

  struct kernel_tripletCleaner {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  TrackingRecHit2DSOAView const *__restrict__ hhp,
                                  HitContainer const *__restrict__ ptuples,
                                  TkSoA const *__restrict__ ptracks,
                                  Quality *__restrict__ quality,
                                  CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ phitToTuple) const {
      constexpr auto bad = trackQuality::bad;
      constexpr auto dup = trackQuality::dup;
      // constexpr auto loose = trackQuality::loose;

      auto &hitToTuple = *phitToTuple;
      auto const &foundNtuplets = *ptuples;
      auto const &tracks = *ptracks;

      //  auto const & hh = *hhp;
      // auto l1end = hh.hitsLayerStart_d[1];

      cms::alpakatools::for_each_element_in_grid_strided(acc, phitToTuple->nbins(), [&](uint32_t idx) {
        if (hitToTuple.size(idx) >= 2) {
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

          if (maxNh <= 3) {
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

          }  // maxNh
        }    // hitToTuple.size
      });    // loop over hits
    }
  };

  struct kernel_print_found_ntuplets {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc,
                                  TrackingRecHit2DSOAView const *__restrict__ hhp,
                                  HitContainer const *__restrict__ ptuples,
                                  TkSoA const *__restrict__ ptracks,
                                  Quality const *__restrict__ quality,
                                  CAHitNtupletGeneratorKernels::HitToTuple const *__restrict__ phitToTuple,
                                  uint32_t maxPrint,
                                  int iev) const {
      auto const &foundNtuplets = *ptuples;
      auto const &tracks = *ptracks;
      const auto np = std::min(maxPrint, foundNtuplets.nbins());
      cms::alpakatools::for_each_element_in_grid_strided(acc, np, [&](uint32_t i) {
        auto nh = foundNtuplets.size(i);
        if (nh >= 3) {
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
        }  // nh
      });
    }
  };

  struct kernel_printCounters {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc &acc, cAHitNtupletGenerator::Counters const *counters) const {
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
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
