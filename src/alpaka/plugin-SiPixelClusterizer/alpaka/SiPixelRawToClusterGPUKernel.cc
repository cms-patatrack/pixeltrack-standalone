/* Sushil Dubey, Shashi Dugad, TIFR, July 2017
 *
 * File Name: RawToClusterGPU.cu
 * Description: It converts Raw data into Digi Format on GPU
 * Finaly the Output of RawToDigi data is given to pixelClusterizer
 *
**/

// C++ includes
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// Alpaka includes
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/prefixScan.h"

// CMSSW includes
#include "gpuCalibPixel.h"
#include "gpuClusterChargeCut.h"
#include "gpuClustering.h"

// local includes
#include "SiPixelRawToClusterGPUKernel.h"




namespace ALPAKA_ACCELERATOR_NAMESPACE {
namespace pixelgpudetails {

  SiPixelRawToClusterGPUKernel::WordFedAppender::WordFedAppender() :
    word_{cms::alpakatools::allocHostBuf<unsigned int>(host, MAX_FED_WORDS)},
    fedId_{cms::alpakatools::allocHostBuf<unsigned char>(host, MAX_FED_WORDS)}
{}

  void SiPixelRawToClusterGPUKernel::WordFedAppender::initializeWordFed(int fedId,
                                                                        unsigned int wordCounterGPU,
                                                                        const uint32_t *src,
                                                                        unsigned int length) {
    std::memcpy(alpaka::getPtrNative(word_) + wordCounterGPU, src, sizeof(uint32_t) * length);
    std::memset(alpaka::getPtrNative(fedId_) + wordCounterGPU / 2, fedId - 1200, length / 2);
  }

  ////////////////////

  ALPAKA_FN_ACC uint32_t getLink(uint32_t ww) {
    return ((ww >> ::pixelgpudetails::LINK_shift) & ::pixelgpudetails::LINK_mask);
  }

  ALPAKA_FN_ACC uint32_t getRoc(uint32_t ww) { return ((ww >> ::pixelgpudetails::ROC_shift) & ::pixelgpudetails::ROC_mask); }

  ALPAKA_FN_ACC uint32_t getADC(uint32_t ww) { return ((ww >> ::pixelgpudetails::ADC_shift) & ::pixelgpudetails::ADC_mask); }

  ALPAKA_FN_ACC bool isBarrel(uint32_t rawId) { return (1 == ((rawId >> 25) & 0x7)); }

  ALPAKA_FN_ACC ::pixelgpudetails::DetIdGPU getRawId(const SiPixelFedCablingMapGPU *cablingMap,
                                                uint8_t fed,
                                                uint32_t link,
                                                uint32_t roc) {
    using namespace ::pixelgpudetails;
    uint32_t index = fed * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
    ::pixelgpudetails::DetIdGPU detId = {
        cablingMap->RawId[index], cablingMap->rocInDet[index], cablingMap->moduleId[index]};
    return detId;
  }

  //reference http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/dd/d31/FrameConversion_8cc_source.html
  //http://cmslxr.fnal.gov/source/CondFormats/SiPixelObjects/src/PixelROC.cc?v=CMSSW_9_2_0#0071
  // Convert local pixel to ::pixelgpudetails::global pixel
  ALPAKA_FN_ACC ::pixelgpudetails::Pixel frameConversion(
      bool bpix, int side, uint32_t layer, uint32_t rocIdInDetUnit, ::pixelgpudetails::Pixel local) {
    int slopeRow = 0, slopeCol = 0;
    int rowOffset = 0, colOffset = 0;

    if (bpix) {
      if (side == -1 && layer != 1) {  // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * ::pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * ::pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * ::pixelgpudetails::numColsInRoc;
        }       // if roc
      } else {  // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
        if (rocIdInDetUnit < 8) {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * ::pixelgpudetails::numRowsInRoc - 1;
          colOffset = rocIdInDetUnit * ::pixelgpudetails::numColsInRoc;
        } else {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (16 - rocIdInDetUnit) * ::pixelgpudetails::numColsInRoc - 1;
        }
      }

    } else {             // fpix
      if (side == -1) {  // pannel 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * ::pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * ::pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * ::pixelgpudetails::numColsInRoc;
        }
      } else {  // pannel 2
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * ::pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * ::pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * ::pixelgpudetails::numColsInRoc;
        }

      }  // side
    }

    uint32_t gRow = rowOffset + slopeRow * local.row;
    uint32_t gCol = colOffset + slopeCol * local.col;
    //printf("Inside frameConversion row: %u, column: %u\n", gRow, gCol);
    ::pixelgpudetails::Pixel global = {gRow, gCol};
    return global;
  }

  ALPAKA_FN_ACC uint8_t conversionError(uint8_t fedId, uint8_t status, bool debug = false) {
    uint8_t errorType = 0;

    // debug = true;

    switch (status) {
      case (1): {
        if (debug)
          printf("Error in Fed: %i, invalid channel Id (errorType = 35\n)", fedId);
        errorType = 35;
        break;
      }
      case (2): {
        if (debug)
          printf("Error in Fed: %i, invalid ROC Id (errorType = 36)\n", fedId);
        errorType = 36;
        break;
      }
      case (3): {
        if (debug)
          printf("Error in Fed: %i, invalid dcol/pixel value (errorType = 37)\n", fedId);
        errorType = 37;
        break;
      }
      case (4): {
        if (debug)
          printf("Error in Fed: %i, dcol/pixel read out of order (errorType = 38)\n", fedId);
        errorType = 38;
        break;
      }
      default:
        if (debug)
          printf("Cabling check returned unexpected result, status = %i\n", status);
    };

    return errorType;
  }

  ALPAKA_FN_ACC bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol) {
    uint32_t numRowsInRoc = 80;
    uint32_t numColsInRoc = 52;

    /// row and collumn in ROC representation
    return ((rocRow < numRowsInRoc) & (rocCol < numColsInRoc));
  }

  ALPAKA_FN_ACC bool dcolIsValid(uint32_t dcol, uint32_t pxid) { return ((dcol < 26) & (2 <= pxid) & (pxid < 162)); }

  ALPAKA_FN_ACC uint8_t checkROC(
      uint32_t errorWord, uint8_t fedId, uint32_t link, const SiPixelFedCablingMapGPU *cablingMap, bool debug = false) {
    using namespace ::pixelgpudetails;
    uint8_t errorType = (errorWord >> ::pixelgpudetails::ROC_shift) & ::pixelgpudetails::ERROR_mask;
    if (errorType < 25)
      return 0;
    bool errorFound = false;

    switch (errorType) {
      case (25): {
        errorFound = true;
        uint32_t index = fedId * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + 1;
        if (index > 1 && index <= cablingMap->size) {
          if (!(link == cablingMap->link[index] && 1 == cablingMap->roc[index]))
            errorFound = false;
        }
        if (debug and errorFound)
          printf("Invalid ROC = 25 found (errorType = 25)\n");
        break;
      }
      case (26): {
        if (debug)
          printf("Gap word found (errorType = 26)\n");
        errorFound = true;
        break;
      }
      case (27): {
        if (debug)
          printf("Dummy word found (errorType = 27)\n");
        errorFound = true;
        break;
      }
      case (28): {
        if (debug)
          printf("Error fifo nearly full (errorType = 28)\n");
        errorFound = true;
        break;
      }
      case (29): {
        if (debug)
          printf("Timeout on a channel (errorType = 29)\n");
        if ((errorWord >> ::pixelgpudetails::OMIT_ERR_shift) & ::pixelgpudetails::OMIT_ERR_mask) {
          if (debug)
            printf("...first errorType=29 error, this gets masked out\n");
        }
        errorFound = true;
        break;
      }
      case (30): {
        if (debug)
          printf("TBM error trailer (errorType = 30)\n");
        int StateMatch_bits = 4;
        int StateMatch_shift = 8;
        uint32_t StateMatch_mask = ~(~uint32_t(0) << StateMatch_bits);
        int StateMatch = (errorWord >> StateMatch_shift) & StateMatch_mask;
        if (StateMatch != 1 && StateMatch != 8) {
          if (debug)
            printf("FED error 30 with unexpected State Bits (errorType = 30)\n");
        }
        if (StateMatch == 1)
          errorType = 40;  // 1=Overflow -> 40, 8=number of ROCs -> 30
        errorFound = true;
        break;
      }
      case (31): {
        if (debug)
          printf("Event number error (errorType = 31)\n");
        errorFound = true;
        break;
      }
      default:
        errorFound = false;
    };

    return errorFound ? errorType : 0;
  }

  ALPAKA_FN_ACC uint32_t getErrRawID(uint8_t fedId,
                                  uint32_t errWord,
                                  uint32_t errorType,
                                  const SiPixelFedCablingMapGPU *cablingMap,
                                  bool debug = false) {
    uint32_t rID = 0xffffffff;

    switch (errorType) {
      case 25:
      case 30:
      case 31:
      case 36:
      case 40: {
        //set dummy values for cabling just to get detId from link
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = 1;
        uint32_t link = (errWord >> ::pixelgpudetails::LINK_shift) & ::pixelgpudetails::LINK_mask;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      case 29: {
        int chanNmbr = 0;
        const int DB0_shift = 0;
        const int DB1_shift = DB0_shift + 1;
        const int DB2_shift = DB1_shift + 1;
        const int DB3_shift = DB2_shift + 1;
        const int DB4_shift = DB3_shift + 1;
        const uint32_t DataBit_mask = ~(~uint32_t(0) << 1);

        int CH1 = (errWord >> DB0_shift) & DataBit_mask;
        int CH2 = (errWord >> DB1_shift) & DataBit_mask;
        int CH3 = (errWord >> DB2_shift) & DataBit_mask;
        int CH4 = (errWord >> DB3_shift) & DataBit_mask;
        int CH5 = (errWord >> DB4_shift) & DataBit_mask;
        int BLOCK_bits = 3;
        int BLOCK_shift = 8;
        uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
        int BLOCK = (errWord >> BLOCK_shift) & BLOCK_mask;
        int localCH = 1 * CH1 + 2 * CH2 + 3 * CH3 + 4 * CH4 + 5 * CH5;
        if (BLOCK % 2 == 0)
          chanNmbr = (BLOCK / 2) * 9 + localCH;
        else
          chanNmbr = ((BLOCK - 1) / 2) * 9 + 4 + localCH;
        if ((chanNmbr < 1) || (chanNmbr > 36))
          break;  // signifies unexpected result

        // set dummy values for cabling just to get detId from link if in Barrel
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = 1;
        uint32_t link = chanNmbr;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      case 37:
      case 38: {
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = (errWord >> ::pixelgpudetails::ROC_shift) & ::pixelgpudetails::ROC_mask;
        uint32_t link = (errWord >> ::pixelgpudetails::LINK_shift) & ::pixelgpudetails::LINK_mask;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      default:
        break;
    };

    return rID;
  }

  // Kernel to perform Raw to Digi conversion
  struct RawToDigi_kernel {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc& acc,
				  const SiPixelFedCablingMapGPU *cablingMap,
				  const unsigned char *modToUnp,
				  const uint32_t wordCounter,
				  const uint32_t *word,
				  const uint8_t *fedIds,
				  uint16_t *xx,
				  uint16_t *yy,
				  uint16_t *adc,
				  uint32_t *pdigi,
				  uint32_t *rawIdArr,
				  uint16_t *moduleId,
				  cms::alpakatools::SimpleVector<PixelErrorCompact> *err,
				  bool useQualityInfo,
				  bool includeErrors,
				  bool debug) const {
      //int32_t first = threadIdx.x + blockIdx.x * blockDim.x;
      //for (int32_t iloop = first, nend = wordCounter; iloop < nend; iloop += blockDim.x * gridDim.x) {


      // TO DO: could also just directly use this function, and replace the continue by return statements.
      // cms::alpakatools::for_each_element_1D_grid_stride(acc, wordCounter, [&](uint32_t iloop) {

      const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u]);
      auto elementIdxShift = Vec1::all(0u);
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] = cms::alpakatools::element_index_range_in_grid(acc, elementIdxShift);
      uint32_t firstElementIdx = firstElementIdxNoStride[0u];
      uint32_t endElementIdx = endElementIdxNoStride[0u];

      for (uint32_t iloop = firstElementIdx; iloop < wordCounter; ++iloop) {
	if (!cms::alpakatools::get_next_element_1D_index_stride(iloop, firstElementIdx, endElementIdx, gridDimension, wordCounter)) { break; }

	auto gIndex = iloop;
	xx[gIndex] = 0;
	yy[gIndex] = 0;
	adc[gIndex] = 0;
	bool skipROC = false;

	uint8_t fedId = fedIds[gIndex / 2];  // +1200;

	// initialize (too many coninue below)
	pdigi[gIndex] = 0;
	rawIdArr[gIndex] = 0;
	moduleId[gIndex] = 9999;

	uint32_t ww = word[gIndex];  // Array containing 32 bit raw data
	if (ww == 0) {
	  // 0 is an indicator of a noise/dead channel, skip these pixels during clusterization
	  continue;
	}

      uint32_t link = getLink(ww);  // Extract link
      uint32_t roc = getRoc(ww);    // Extract Roc in link
      ::pixelgpudetails::DetIdGPU detId = getRawId(cablingMap, fedId, link, roc);

      uint8_t errorType = checkROC(ww, fedId, link, cablingMap, debug);
      skipROC = (roc < ::pixelgpudetails::maxROCIndex) ? false : (errorType != 0);
      if (includeErrors and skipROC) {
        uint32_t rID = getErrRawID(fedId, ww, errorType, cablingMap, debug);
        err->push_back(acc, PixelErrorCompact{rID, ww, errorType, fedId});
        continue;
      }

      uint32_t rawId = detId.RawId;
      uint32_t rocIdInDetUnit = detId.rocInDet;
      bool barrel = isBarrel(rawId);

      uint32_t index = fedId * ::pixelgpudetails::MAX_LINK * ::pixelgpudetails::MAX_ROC + (link - 1) * ::pixelgpudetails::MAX_ROC + roc;
      if (useQualityInfo) {
        skipROC = cablingMap->badRocs[index];
        if (skipROC)
          continue;
      }
      skipROC = modToUnp[index];
      if (skipROC)
        continue;

      uint32_t layer = 0;                   //, ladder =0;
      int side = 0, panel = 0, module = 0;  //disk = 0, blade = 0

      if (barrel) {
        layer = (rawId >> ::pixelgpudetails::layerStartBit) & ::pixelgpudetails::layerMask;
        module = (rawId >> ::pixelgpudetails::moduleStartBit) & ::pixelgpudetails::moduleMask;
        side = (module < 5) ? -1 : 1;
      } else {
        // endcap ids
        layer = 0;
        panel = (rawId >> ::pixelgpudetails::panelStartBit) & ::pixelgpudetails::panelMask;
        //disk  = (rawId >> diskStartBit_) & diskMask_;
        side = (panel == 1) ? -1 : 1;
        //blade = (rawId >> bladeStartBit_) & bladeMask_;
      }

      // ***special case of layer to 1 be handled here
      ::pixelgpudetails::Pixel localPix;
      if (layer == 1) {
        uint32_t col = (ww >> ::pixelgpudetails::COL_shift) & ::pixelgpudetails::COL_mask;
        uint32_t row = (ww >> ::pixelgpudetails::ROW_shift) & ::pixelgpudetails::ROW_mask;
        localPix.row = row;
        localPix.col = col;
        if (includeErrors) {
          if (not rocRowColIsValid(row, col)) {
            uint8_t error = conversionError(fedId, 3, debug);  //use the device function and fill the arrays
            err->push_back(acc, PixelErrorCompact{rawId, ww, error, fedId});
            if (debug)
              printf("BPIX1  Error status: %i\n", error);
            continue;
          }
        }
      } else {
        // ***conversion rules for dcol and pxid
        uint32_t dcol = (ww >> ::pixelgpudetails::DCOL_shift) & ::pixelgpudetails::DCOL_mask;
        uint32_t pxid = (ww >> ::pixelgpudetails::PXID_shift) & ::pixelgpudetails::PXID_mask;
        uint32_t row = ::pixelgpudetails::numRowsInRoc - pxid / 2;
        uint32_t col = dcol * 2 + pxid % 2;
        localPix.row = row;
        localPix.col = col;
        if (includeErrors and not dcolIsValid(dcol, pxid)) {
          uint8_t error = conversionError(fedId, 3, debug);
          err->push_back(acc, PixelErrorCompact{rawId, ww, error, fedId});
          if (debug)
            printf("Error status: %i %d %d %d %d\n", error, dcol, pxid, fedId, roc);
          continue;
        }
      }

      ::pixelgpudetails::Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
      xx[gIndex] = globalPix.row;  // origin shifting by 1 0-159
      yy[gIndex] = globalPix.col;  // origin shifting by 1 0-415
      adc[gIndex] = getADC(ww);
      pdigi[gIndex] = ::pixelgpudetails::pack(globalPix.row, globalPix.col, adc[gIndex]);
      moduleId[gIndex] = detId.moduleId;
      rawIdArr[gIndex] = rawId;
      }  // end of stride on grid
      //});  // end of stride on grid

  }  
  }; // end of Raw to Digi kernel

}    // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace pixelgpudetails {

  struct fillHitsModuleStart {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc& acc,
				  uint32_t const *__restrict__ cluStart, 
				  uint32_t *__restrict__ moduleStart) const {
    assert(gpuClustering::MaxNumModules < 2048);  // easy to extend at least till 32*1024

    const uint32_t blockIdxLocal(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]); 
    assert(0 == blockIdxLocal);
    const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
    assert(1 == gridDimension);

    //int first = threadIdx.x;

    // limit to MaxHitsInModule;
    //for (int i = first, iend = gpuClustering::MaxNumModules; i < iend; i += blockDim.x) {
    cms::alpakatools::for_each_element_1D_block_stride(acc, gpuClustering::MaxNumModules, [&](uint32_t i) {
      moduleStart[i + 1] = std::min(gpuClustering::maxHitsInModule(), cluStart[i]);
      });

    auto&& ws = alpaka::declareSharedVar<uint32_t[32], __COUNTER__>(acc);
    cms::alpakatools::blockPrefixScan(acc, moduleStart + 1, moduleStart + 1, 1024, ws);
    cms::alpakatools::blockPrefixScan(acc, moduleStart + 1025, moduleStart + 1025, gpuClustering::MaxNumModules - 1024, ws);

    //for (int i = first + 1025, iend = gpuClustering::MaxNumModules + 1; i < iend; i += blockDim.x) {
    cms::alpakatools::for_each_element_1D_block_stride(acc, gpuClustering::MaxNumModules + 1, 1025u, [&](uint32_t i) {
	moduleStart[i] += moduleStart[1024];
      });
    alpaka::syncBlockThreads(acc);

#ifdef GPU_DEBUG
    assert(0 == moduleStart[0]);
    auto c0 = std::min(gpuClustering::maxHitsInModule(), cluStart[0]);
    assert(c0 == moduleStart[1]);
    assert(moduleStart[1024] >= moduleStart[1023]);
    assert(moduleStart[1025] >= moduleStart[1024]);
    assert(moduleStart[gpuClustering::MaxNumModules] >= moduleStart[1025]);

      //for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += blockDim.x) {
      cms::alpakatools::for_each_element_1D_block_stride(acc, gpuClustering::MaxNumModules + 1, [&](uint32_t i) {
      if (0 != i)
        assert(moduleStart[i] >= moduleStart[i - i]);
      // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
      // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
      if (i == 96 || i == 1184 || i == 1744 || i == gpuClustering::MaxNumModules)
        printf("moduleStart %d %d\n", i, moduleStart[i]);
	});
#endif

    // avoid overflow
    constexpr auto MAX_HITS = gpuClustering::MaxNumClusters;
    //for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += blockDim.x) {
    cms::alpakatools::for_each_element_1D_block_stride(acc, gpuClustering::MaxNumModules + 1, [&](uint32_t i) {
      if (moduleStart[i] > MAX_HITS)
        moduleStart[i] = MAX_HITS;
      });

    }
  }; // end of fillHitsModuleStart kernel

}  // namespace pixelgpudetails

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelgpudetails {

  // Interface to outside
  void SiPixelRawToClusterGPUKernel::makeClustersAsync(bool isRun2,
                                                       const SiPixelFedCablingMapGPU *cablingMap,
                                                       const unsigned char *modToUnp,
                                                       const SiPixelGainForHLTonGPU *gains,
                                                       const WordFedAppender &wordFed,
                                                       PixelFormatterErrors &&errors,
                                                       const uint32_t wordCounter,
                                                       const uint32_t fedCounter,
                                                       bool useQualityInfo,
                                                       bool includeErrors,
                                                       bool debug,
                                                       Queue& queue) {
    nDigis = wordCounter;

    std::cout << "wordCounter = " << wordCounter << std::endl;

#ifdef GPU_DEBUG
    std::cout << "decoding " << wordCounter << " digis. Max is " << pixelgpudetails::MAX_FED_WORDS << std::endl;
#endif

    //digis_d = SiPixelDigisAlpaka(pixelgpudetails::MAX_FED_WORDS);
    if (includeErrors) {
      //digiErrors_d = SiPixelDigiErrorsAlpaka(pixelgpudetails::MAX_FED_WORDS, std::move(errors));
      digiErrors_d.setFormatterErrors(std::move(errors));
    }
    //clusters_d = SiPixelClustersAlpaka(gpuClustering::MaxNumModules);

    if (wordCounter)  // protect in case of empty event....
    {
      
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      const int threadsPerBlockOrElementsPerThread = 512;    
#else
      // This should be tuned.
      const int threadsPerBlockOrElementsPerThread = 1;
#endif
      const uint32_t blocks = (wordCounter + threadsPerBlockOrElementsPerThread - 1) / threadsPerBlockOrElementsPerThread;  // fill it all
      const WorkDiv1& workDiv = cms::alpakatools::make_workdiv(Vec1::all(blocks),
	Vec1::all(threadsPerBlockOrElementsPerThread));

      assert(0 == wordCounter % 2);
      // wordCounter is the total no of words in each event to be trasfered on device
      auto word_d = cms::alpakatools::allocDeviceBuf<uint32_t>(device, wordCounter);
      auto fedId_d = cms::alpakatools::allocDeviceBuf<uint8_t>(device, wordCounter);

      // TO DO: NOOOOO different size between src and det, so might not work!!
      alpaka::memcpy(queue, word_d, wordFed.word(), wordCounter);
      alpaka::memcpy(queue, fedId_d, wordFed.fedId(), wordCounter/2); // TO DO: wordCounter/2?

      /*auto wordFedWordView = cms::alpakatools::createDeviceView<uint32_t>(device, wordFed.word(), MAX_FED_WORDS);
      SubView<uint32_t> wordFedWordSubView = SubView<uint32_t>(wordFedWordView, wordCounter);
      auto wordFedFedIdView = cms::alpakatools::createDeviceView<uint8_t>(device, wordFed.word(), MAX_FED_WORDS);
      SubView<uint8_t> wordFedFedIdSubView = SubView<uint8_t>(wordFedFedIdView, wordCounter/2);
      alpaka::memcpy(queue, word_d, wordFedWordSubView, wordCounter);
      alpaka::memcpy(queue, fedId_d, wordFedFedIdSubView, wordCounter/2); // TO DO: wordCounter/2?*/

      // Launch rawToDigi kernel
      alpaka::enqueue(queue,
        alpaka::createTaskKernel<Acc1>(workDiv,
	RawToDigi_kernel(),
	cablingMap,
	modToUnp,
	wordCounter,
	alpaka::getPtrNative(word_d),
	alpaka::getPtrNative(fedId_d),
	digis_d.xx(),
	digis_d.yy(),
	digis_d.adc(),
	digis_d.pdigi(),
	digis_d.rawIdArr(),
	digis_d.moduleInd(),
	digiErrors_d.error(),  // returns nullptr if default-constructed
	useQualityInfo,
	includeErrors,
	debug));
     
#ifdef GPU_DEBUG
      alpaka::wait(device);
#endif

#ifdef TODO
      if (includeErrors) {
        digiErrors_d.copyErrorToHostAsync(stream);
      }
#endif
    }
    // End of Raw2Digi and passing data for clustering

    {
      // clusterizer ...
      using namespace gpuClustering;	
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      int threadsPerBlockOrElementsPerThread = 256;    
#else
      int threadsPerBlockOrElementsPerThread = 1; // This should be tuned.
#endif
      int blocks = (std::max(int(wordCounter), int(gpuClustering::MaxNumModules)) + threadsPerBlockOrElementsPerThread - 1) / threadsPerBlockOrElementsPerThread;
      const WorkDiv1& workDiv = cms::alpakatools::make_workdiv(Vec1::all(blocks),
	Vec1::all(threadsPerBlockOrElementsPerThread));

      alpaka::enqueue(queue,
		      alpaka::createTaskKernel<Acc1>(workDiv,
						     gpuCalibPixel::calibDigis(),
						     isRun2,
						     digis_d.moduleInd(),
						     digis_d.c_xx(),
						     digis_d.c_yy(),
						     digis_d.adc(),
						     //gains,
						     gains->getVpedestals(),
						     gains->getRangeAndCols(),
						     gains->getFields(),
						     wordCounter,
						     clusters_d.moduleStart(),
						     clusters_d.clusInModule(),
						     clusters_d.clusModuleStart()
						     ));
 
#ifdef GPU_DEBUG
      alpaka::wait(device);
#endif

#ifdef GPU_DEBUG
      std::cout << "CUDA countModules kernel launch with " << blocks << " blocks of " << threadsPerBlockOrElementsPerThread
                << " threadsPerBlockOrElementsPerThread\n";
#endif

      alpaka::enqueue(queue,
	alpaka::createTaskKernel<Acc1>(workDiv,
	countModules(),
	digis_d.c_moduleInd(), 
	clusters_d.moduleStart(), 
	digis_d.clus(), 
	wordCounter
	));
 

      // TO DO: NOOOOO clusters_d.moduleStart() IS NOT A BUFFER but a raw pointer!!!!!!!!!
      //cudaCheck(cudaMemcpyAsync(
      //&(nModules_Clusters_h[0]), clusters_d.moduleStart(), sizeof(uint32_t), cudaMemcpyDefault, stream));

      // read the number of modules into a data member, used by getProduct())
      auto nModules_Clusters_0_h{cms::alpakatools::allocHostBuf<uint32_t>(host, 1u)};
      auto p_nModules_Clusters_0_h = alpaka::getPtrNative(nModules_Clusters_0_h);

      auto moduleStartView = cms::alpakatools::createDeviceView<uint32_t>(device, clusters_d.moduleStart(), gpuClustering::MaxNumModules + 1);
      SubView<uint32_t> moduleStartSubView = SubView<uint32_t>(moduleStartView, 1u, 0u);
      //alpaka::memcpy(queue, nModules_Clusters_0_h, clusters_d.moduleStart(), 1u);
      alpaka::memcpy(queue, nModules_Clusters_0_h, moduleStartSubView, 1u);
      
      auto p_nModules_Clusters_h = alpaka::getPtrNative(nModules_Clusters_h);
      p_nModules_Clusters_h[0] = p_nModules_Clusters_0_h[0];


      //threadsPerBlock = 256;
      blocks = MaxNumModules;
      const WorkDiv1& workDivMaxNumModules = cms::alpakatools::make_workdiv(Vec1::all(blocks),
	Vec1::all(threadsPerBlockOrElementsPerThread));

#ifdef GPU_DEBUG
      std::cout << "CUDA findClus kernel launch with " << blocks << " blocks of " << threadsPerBlockOrElementsPerThread << " threadsPerBlockOrElementsPerThread\n";
#endif

      alpaka::enqueue(queue,
	alpaka::createTaskKernel<Acc1>(workDivMaxNumModules,
	findClus(),
	digis_d.c_moduleInd(),
	digis_d.c_xx(),
	digis_d.c_yy(),
	clusters_d.c_moduleStart(),
	clusters_d.clusInModule(),
	clusters_d.moduleId(),
	digis_d.clus(),
	wordCounter));
   
#ifdef GPU_DEBUG
      //alpaka::wait(device);
#endif

      // apply charge cut
      alpaka::enqueue(queue,
	alpaka::createTaskKernel<Acc1>(workDivMaxNumModules,
	clusterChargeCut(),
	digis_d.moduleInd(),
	digis_d.c_adc(),
	clusters_d.c_moduleStart(),
	clusters_d.clusInModule(),
	clusters_d.c_moduleId(),
	digis_d.clus(),
	wordCounter));

      // count the module start indices already here (instead of
      // rechits) so that the number of clusters/hits can be made
      // available in the rechit producer without additional points of
      // synchronization/ExternalWork

      threadsPerBlockOrElementsPerThread = 1024;
      blocks = 1;
      const WorkDiv1& workDivOneBlock = cms::alpakatools::make_workdiv(Vec1::all(blocks),
	Vec1::all(threadsPerBlockOrElementsPerThread));

      // MUST be ONE block
      alpaka::enqueue(queue,
		      alpaka::createTaskKernel<Acc1>(workDivOneBlock,
						     ::pixelgpudetails::fillHitsModuleStart(),
						     clusters_d.c_clusInModule(), 
						     clusters_d.clusModuleStart()
						     ));
      
      // last element holds the number of all clusters  

      // slice on host
      auto nModules_Clusters_1_h{cms::alpakatools::allocHostBuf<uint32_t>(host, 1u)};
      auto p_nModules_Clusters_1_h = alpaka::getPtrNative(nModules_Clusters_1_h);

      auto clusModuleStartView = cms::alpakatools::createDeviceView<uint32_t>(device, clusters_d.clusModuleStart(), gpuClustering::MaxNumModules + 1);
      SubView<uint32_t> clusModuleStartSubView = SubView<uint32_t>(clusModuleStartView, 1u, gpuClustering::MaxNumModules);
 
      alpaka::memcpy(queue, nModules_Clusters_1_h, clusModuleStartSubView, 1u);
      p_nModules_Clusters_h[1] = p_nModules_Clusters_1_h[0];


#ifdef GPU_DEBUG
      //alpaka::wait(device);
#endif

      alpaka::wait(queue);
    }  // end clusterizer scope
  }
}  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
