#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h

#define CONSTANT_VAR __constant__

namespace gpuPixelDoublets {

  constexpr int nPairs = 13 + 2 + 4;
  static_assert(nPairs <= CAConstants::maxNumberOfLayerPairs());

  // start constants
  // clang-format off

  CONSTANT_VAR const uint8_t layerPairs[2 * nPairs] = {
      0, 1, 0, 4, 0, 7,              // BPIX1 (3)
      1, 2, 1, 4, 1, 7,              // BPIX2 (5)
      4, 5, 7, 8,                    // FPIX1 (8)
      2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
      0, 2, 1, 3,                    // Jumping Barrel (15)
      0, 5, 0, 8,                    // Jumping Forward (BPIX1,FPIX2)
      4, 6, 7, 9                     // Jumping Forward (19)
  };

  constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);

  CONSTANT_VAR const int16_t phicuts[nPairs]{phi0p05,
                                             phi0p07,
                                             phi0p07,
                                             phi0p05,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p06,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05};
  //   phi0p07, phi0p07, phi0p06,phi0p06, phi0p06,phi0p06};  // relaxed cuts

  CONSTANT_VAR float const minz[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  CONSTANT_VAR float const maxz[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  CONSTANT_VAR float const maxr[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

  // end constants
  // clang-format on

  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __global__ void initDoublets(GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                               int nHits,
                               CellNeighborsVector* cellNeighbors,
                               CellNeighbors* cellNeighborsContainer,
                               CellTracksVector* cellTracks,
                               CellTracks* cellTracksContainer) {
    assert(isOuterHitOfCell);
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = first; i < nHits; i += gridDim.x * blockDim.x)
      isOuterHitOfCell[i].reset();

    if (0 == first) {
      cellNeighbors->construct(CAConstants::maxNumOfActiveDoublets(), cellNeighborsContainer);
      cellTracks->construct(CAConstants::maxNumOfActiveDoublets(), cellTracksContainer);
      auto i = cellNeighbors->extend();
      assert(0 == i);
      (*cellNeighbors)[0].reset();
      i = cellTracks->extend();
      assert(0 == i);
      (*cellTracks)[0].reset();
    }
  }

  constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
  constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

  __global__
#if defined(__NVCOMPILER) || defined(__CUDACC__)
  __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)
#endif
      void getDoubletsFromHisto(GPUCACell* cells,
                                uint32_t* nCells,
                                CellNeighborsVector* cellNeighbors,
                                CellTracksVector* cellTracks,
                                TrackingRecHit2DSOAView const* __restrict__ hhp,
                                GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                int nPairs,
                                bool ideal_cond,
                                bool doClusterCut,
                                bool doZ0Cut,
                                bool doPtCut,
                                uint32_t maxNumOfDoublets) {
    auto const& __restrict__ hh = *hhp;
    // ysize cuts (z in the barrel)  times 8
    // these are used if doClusterCut is true
    constexpr int minYsizeB1 = 36;
    constexpr int minYsizeB2 = 28;
    constexpr int maxDYsize12 = 28;
    constexpr int maxDYsize = 20;
    constexpr int maxDYPred = 20;
    constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"

    bool isOuterLadder = ideal_cond;

    using Hist = TrackingRecHit2DSOAView::Hist;

    auto const& __restrict__ hist = hh.phiBinner();
    uint32_t const* __restrict__ offsets = hh.hitsLayerStart();
    assert(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

    // nPairsMax to be optimized later (originally was 64).
    // If it should be much bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    const int nPairsMax = CAConstants::maxNumberOfLayerPairs();
    assert(nPairs <= nPairsMax);
    __shared__ uint32_t innerLayerCumulativeSize[nPairsMax];
    __shared__ uint32_t ntot;
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      innerLayerCumulativeSize[0] = layerSize(layerPairs[0]);
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(layerPairs[2 * i]);
      }
      ntot = innerLayerCumulativeSize[nPairs - 1];
    }
    __syncthreads();

    // x runs faster
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;

    uint32_t pairLayerId = 0;  // cannot go backward
    for (auto j = idy; j < ntot; j += blockDim.y * gridDim.y) {
      while (j >= innerLayerCumulativeSize[pairLayerId++])
        ;
      --pairLayerId;  // move to lower_bound ??

      assert(pairLayerId < nPairs);
      assert(j < innerLayerCumulativeSize[pairLayerId]);
      assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

      uint8_t inner = layerPairs[2 * pairLayerId];
      uint8_t outer = layerPairs[2 * pairLayerId + 1];
      assert(outer > inner);

      auto hoff = Hist::histOff(outer);

      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      i += offsets[inner];

      // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

      assert(i >= offsets[inner]);
      assert(i < offsets[inner + 1]);

      // found hit corresponding to our cuda thread, now do the job
      auto mi = hh.detectorIndex(i);
      if (mi > 2000)
        continue;  // invalid

      /* maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */

      auto mez = hh.zGlobal(i);

      if (mez < minz[pairLayerId] || mez > maxz[pairLayerId])
        continue;

      int16_t mes = -1;  // make compiler happy
      if (doClusterCut) {
        // if ideal treat inner ladder as outer
        if (inner == 0)
          assert(mi < 96);
        isOuterLadder = ideal_cond ? true : 0 == (mi / 8) % 2;  // only for B1/B2/B3 B4 is opposite, FPIX:noclue...

        // in any case we always test mes>0 ...
        mes = inner > 0 || isOuterLadder ? hh.clusterSizeY(i) : -1;

        if (inner == 0 && outer > 3)  // B1 and F1
          if (mes > 0 && mes < minYsizeB1)
            continue;                 // only long cluster  (5*8)
        if (inner == 1 && outer > 3)  // B2 and F1
          if (mes > 0 && mes < minYsizeB2)
            continue;
      }
      auto mep = hh.iphi(i);
      auto mer = hh.rGlobal(i);

      // all cuts: true if fails
      constexpr float z0cut = 12.f;      // cm
      constexpr float hardPtCut = 0.5f;  // GeV
      constexpr float minRadius =
          hardPtCut * 87.78f;  // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
      constexpr float minRadius2T4 = 4.f * minRadius * minRadius;
      auto ptcut = [&](int j, int16_t idphi) {
        auto r2t4 = minRadius2T4;
        auto ri = mer;
        auto ro = hh.rGlobal(j);
        auto dphi = short2phi(idphi);
        return dphi * dphi * (r2t4 - ri * ro) > (ro - ri) * (ro - ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        auto dr = ro - mer;
        return dr > maxr[pairLayerId] || dr < 0 || std::abs((mez * ro - mer * zo)) > z0cut * dr;
      };

      auto zsizeCut = [&](int j) {
        auto onlyBarrel = outer < 4;
        auto so = hh.clusterSizeY(j);
        auto dy = inner == 0 ? maxDYsize12 : maxDYsize;
        // in the barrel cut on difference in size
        // in the endcap on the prediction on the first layer (actually in the barrel only: happen to be safe for endcap as well)
        // FIXME move pred cut to z0cutoff to optmize loading of and computaiton ...
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        return onlyBarrel ? mes > 0 && so > 0 && std::abs(so - mes) > dy
                          : (inner < 4) && mes > 0 &&
                                std::abs(mes - int(std::abs((mez - zo) / (mer - ro)) * dzdrFact + 0.5f)) > maxDYPred;
      };

      auto iphicut = phicuts[pairLayerId];

      auto kl = Hist::bin(int16_t(mep - iphicut));
      auto kh = Hist::bin(int16_t(mep + iphicut));
      auto incr = [](auto& k) { return k = (k + 1) % Hist::nbins(); };
      // bool piWrap = std::abs(kh-kl) > Hist::nbins()/2;

#ifdef GPU_DEBUG
      int tot = 0;
      int nmin = 0;
      int tooMany = 0;
#endif

      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
#ifdef GPU_DEBUG
        if (kk != kl && kk != kh)
          nmin += hist.size(kk + hoff);
#endif
        auto const* __restrict__ p = hist.begin(kk + hoff);
        auto const* __restrict__ e = hist.end(kk + hoff);
        p += first;
        for (; p < e; p += stride) {
          auto oi = __ldg(p);
          assert(oi >= offsets[outer]);
          assert(oi < offsets[outer + 1]);
          auto mo = hh.detectorIndex(oi);
          if (mo > 2000)
            continue;  //    invalid

          if (doZ0Cut && z0cutoff(oi))
            continue;

          auto mop = hh.iphi(oi);
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));
          if (idphi > iphicut)
            continue;

          if (doClusterCut && zsizeCut(oi))
            continue;
          if (doPtCut && ptcut(oi, idphi))
            continue;

          auto ind = atomicAdd(nCells, 1);
          if (ind >= maxNumOfDoublets) {
            atomicSub(nCells, 1);
            break;
          }  // move to SimpleVector??
          // int layerPairId, int doubletId, int innerHitId, int outerHitId)
          cells[ind].init(*cellNeighbors, *cellTracks, hh, pairLayerId, ind, i, oi);
          isOuterHitOfCell[oi].push_back(ind);
#ifdef GPU_DEBUG
          if (isOuterHitOfCell[oi].full())
            ++tooMany;
          ++tot;
#endif
        }
      }
#ifdef GPU_DEBUG
      if (tooMany > 0)
        printf("OuterHitOfCell full for %d in layer %d/%d, %d,%d %d\n", i, inner, outer, nmin, tot, tooMany);
#endif
    }  // loop in block...
  }

}  // namespace gpuPixelDoublets

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
