#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoubletsAlgos_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoubletsAlgos_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <algorithm>

#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "DataFormats/approx_atan2.h"
#include "CUDACore/VecArray.h"
#include "CUDACore/cuda_assert.h"

#include "CAConstants.h"
#include "GPUCACell.h"

namespace gpuPixelDoublets {

  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  auto getIndex(uint8_t i, std::vector<uint8_t> layers_) {
    auto it = std::find(layers_.begin(), layers_.end(), i);
    int index = std::distance(layers_.begin(), it);
    return index;
  }

  __device__ __forceinline__ void doubletsFromHisto(uint8_t const* __restrict__ layerPairs,
                                                    uint32_t nPairs,
                                                    GPUCACell* cells,
                                                    uint32_t* nCells,
                                                    CellNeighborsVector* cellNeighbors,
                                                    CellTracksVector* cellTracks,
                                                    TrackingRecHit2DSOAView const& __restrict__ hh,
                                                    GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                                    int16_t const* __restrict__ phicuts,
                                                    float const* __restrict__ minz,
                                                    float const* __restrict__ maxz,
                                                    float const* __restrict__ maxr,
                                                    bool ideal_cond,
                                                    bool doClusterCut,
                                                    bool doZ0Cut,
                                                    bool doPtCut,
                                                    uint32_t maxNumOfDoublets) {
    // ysize cuts (z in the barrel)  times 8
    // these are used if doClusterCut is true
    constexpr int minYsizeB1 = 36;
    constexpr int minYsizeB2 = 28;
    constexpr int maxDYsize12 = 28;
    constexpr int maxDYsize = 20;
    constexpr int maxDYPred = 20;
    constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"

    bool isOuterLadder = ideal_cond;
    //std::cout7 << "isOuterLadder" << isOuterLadder << '\n';

    using Hist = TrackingRecHit2DSOAView::Hist;

    std::cout << "inizio algos" << '\n';
    auto const& __restrict__ hist = hh.phiBinner();
    auto const* offsets = hh.hitsLayerStart();
    for(int j = 0; j < 48; ++j) {
      //std::cout7 << offsets[j] << '\n';
      //std::cout7 << hh.hitsLayerStart()[j] << '\n';
      //if(offsets[j]) {
      //  //std::cout7 << j << '\n';
      //}
    }
    assert(offsets);

    std::vector<uint8_t> layers = {10,9,8,7,6,5,4,        // vol7
                                   0,1,2,3,               // vol8
                                   11,12,13,14,15,16,17,  // vol9
                                   27,26,25,24,23,22,     // vol12
                                   18,19,20,21,           // vol13
                                   28,29,30,31,32,33,     // vol14
                                   41,40,39,38,37,36,     // vol16
                                   34,35,                 // vol17
                                   42,43,44,45,46,47};    // vol18

    //#ifdef NOTRACKML
    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };   // how many hits in that layer
    //#else
    //auto layerSize = [=](uint8_t lay) { return offsets[getIndex(lay,layers)+1] - offsets[getIndex(lay,layers)]; };
    //#endif

    // nPairsMax to be optimized later (originally was 64).
    // If it should be much bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    const int nPairsMax = CAConstants::maxNumberOfLayerPairs();
    assert(nPairs <= nPairsMax);
    __shared__ uint32_t innerLayerCumulativeSize[nPairsMax];
    __shared__ uint32_t ntot;
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      innerLayerCumulativeSize[0] = layerSize(layerPairs[0]);
      //std::cout7 << layerSize(1) << '\n';
      //std::cout7 << layerSize(layerPairs[0]) << '\n';
      ////std::cout7 << "innerLayerCumulativeSize[0] " << innerLayerCumulativeSize[0] << '\n';
      ////std::cout7 << "layerSize(layerPairs[3]) " << layerSize(layerPairs[3]) << '\n';
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(layerPairs[2 * i]);
        ////std::cout7 << "innerLayerCumulativeSize[i]" << i << ' ' << innerLayerCumulativeSize[i] << '\n';
      }
      //#ifdef TEST
      //ntot = innerLayerCumulativeSize[nPairs - 1];
      ntot = 1500;
      //#else
      //ntot = 100000;
      //#endif
    }
    __syncthreads();
    // x runs faster
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;

    uint32_t pairLayerId = 0;  // cannot go backward
    for (auto j = idy; j < ntot; j += blockDim.y * gridDim.y) {
      ////std::cout7 << "j" << j << '\n';
      ////std::cout7 << "innerroba " << innerLayerCumulativeSize[0] << '\n';
      while (j >= innerLayerCumulativeSize[pairLayerId++]) {
        ;
      }
      --pairLayerId;  // move to lower_bound ??
      assert(pairLayerId < nPairs);
      assert(j < innerLayerCumulativeSize[pairLayerId]);
      assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

      uint8_t inner = layerPairs[2 * pairLayerId];
      uint8_t outer = layerPairs[2 * pairLayerId + 1];
      ////std::cout7 << "inner" << unsigned(inner) << '\n';
      //std::cout7 << "outer" << unsigned(outer) << '\n';
      assert(outer > inner);

      auto hoff = Hist::histOff(outer);

      //std::cout7 << "pairid" << pairLayerId << '\n';
      //std::cout7 << "j " << j << '\n';
      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      //std::cout7 << "i before offset" << i << '\n';
      i += offsets[inner];
      //std::cout7 << "offset inner" << offsets[inner] << '\n';
      //std::cout7 << "offset inner + 1" << offsets[inner+1] << '\n';
      //std::cout7 << "i after offset" << i  << '\n';
      // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

      //std::cout7 << "offsets[inner]" << offsets[inner] << '\n';
      //std::cout7 << "offsets[inner+1]" << offsets[inner+1] << '\n';
      assert(i >= offsets[inner]);
      //std::cout7 << "i " << i << '\n';
      assert(i < offsets[inner + 1]);

      // found hit corresponding to our cuda thread, now do the job
      // auto mi = hh.detectorIndex(i);   metti tutti isOuterLadder
      //if (mi > 2000)
      //  continue;  // invalid

      /* maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */
     
      auto mez = hh.zGlobal(i);
      // if (mez < minz[pairLayerId] || mez > maxz[pairLayerId])
      //   continue;

      int16_t mes = -1;  // make compiler happy
      doClusterCut = false;
      if (doClusterCut) {
        // if ideal treat inner ladder as outerhttps://docs.google.com/document/d/15rYOJF7gU2R8XysKd-6_1EQoUYi27hxXDeSokcTIqHk/edit?usp=sharing
        isOuterLadder = true;
        //#endif

        // in any case we always test mes>0 ...
        //#ifdef NOTRACKML
        //mes = inner > 0 || isOuterLadder ? hh.clusterSizeY(i) : -1;
        //#else
        //mes = -1;
        //#endif

        //if (inner == 0 && outer > 3)  // B1 and F1
        //  if (mes > 0 && mes < minYsizeB1)
        //    continue;                 // only long cluster  (5*8)
        //if (inner == 1 && outer > 3)  // B2 and F1
        //  if (mes > 0 && mes < minYsizeB2)
        //    continue;
      }
      auto mep = hh.iphi(i);    // riempire iphi in qualche modo
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
        //auto so = hh.clusterSizeY(j);
        int so = 30;
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
      std::cout << "phi cut " << iphicut << "mep " << mep << std::endl;
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
        std::cout << "kk " << kk << '\n';      // printa 126
        std::cout << "hoff " << hoff << '\n';  // printa 256
        std::cout << hist.bins[0] << '\n';     // printa 0
        //TrackingRecHit2DSOAView::Hist hist2;
        auto const* __restrict__ p = hist.begin(kk + hoff);
        auto const* __restrict__ e = hist.end(kk + hoff);
        //std::cout << "nbins " << hist2.nbins() << '\n';
        //std::cout << "capacity " << hist2.capacity() << '\n';
        //std::cout << "nhists " << hist2.nhists() << '\n';
        //std::cout << "bin0 " << hist2.bin(0) << '\n';
        //std::cout << "binnbins " << hist2.bin(hist2.nbins()) << '\n';
        //std::cout << "tot " << hist2.totbins() << '\n';
        //std::cout << "manual p " <<  hist2.off[kk + hoff] << std::endl;
        //std::cout << "manual e " <<  hist2.off[kk + hoff + 1] << std::endl;
        p += first;
        std::cout << "p,e " << p << ' ' << e << '\n';   // prints 0 and 0, so it doesn't enter the for loop
        //*assert(p<e);
        std::cout << "prima del for" << '\n';

        std::cout << "FIRST " << first << " Stride " << stride << std::endl;
        for (; p < e; p += stride) {
          std::cout << "p,e,stride " << p << ' ' << e << ' ' << stride << '\n';   // see above
          auto oi = __ldg(p);
          assert(oi >= offsets[outer]);
          assert(oi < offsets[outer + 1]);
          //auto mo = hh.detectorIndex(oi);   // what is this?
          //if (mo > 2000)
          //  continue;  //    invalid

          //if (doZ0Cut && z0cutoff(oi))
          //  continue;

          auto mop = hh.iphi(oi);
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));
          //if (idphi > iphicut)
          //  continue;

          //if (doClusterCut && zsizeCut(oi))
          //  continue;
          //if (doPtCut && ptcut(oi, idphi))
          //  continue;

          auto ind = atomicAdd(nCells, 1);
          std::cout << "ind,maxdubl" << ind << ' ' << maxNumOfDoublets << '\n';
          if (ind >= maxNumOfDoublets) {
            atomicSub(nCells, 1);
            break;
          }  // move to SimpleVector??
          // int layerPairId, int doubletId, int innerHitId, int outerHitId)
          cells[ind].init(*cellNeighbors, *cellTracks, hh, pairLayerId, ind, i, oi);
          std::cout << "inner x" << cells[ind].get_inner_x(hh) << '\n';
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
    std::cout << "Fine di doubletsFromHist" << '\n';
  }

}  // namespace gpuPixelDoublets

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoupletsAlgos_h
