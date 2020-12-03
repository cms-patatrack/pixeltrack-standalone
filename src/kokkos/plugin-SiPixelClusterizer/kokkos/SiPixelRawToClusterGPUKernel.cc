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

// CMSSW includes
#include "KokkosCore/hintLightWeight.h"
#include "KokkosDataFormats/gpuClusteringConstants.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include "gpuCalibPixel.h"
#include "gpuClusterChargeCut.h"
#include "gpuClustering.h"

// local includes
#include "SiPixelRawToClusterGPUKernel.h"

namespace KOKKOS_NAMESPACE {
  namespace pixelgpudetails {
    constexpr uint32_t MAX_FED_WORDS = ::pixelgpudetails::MAX_FED * ::pixelgpudetails::MAX_WORD;

    SiPixelRawToClusterGPUKernel::WordFedAppender::WordFedAppender()
        : word_(Kokkos::ViewAllocateWithoutInitializing("word"), MAX_FED_WORDS),
          fedId_(Kokkos::ViewAllocateWithoutInitializing("fedId"), MAX_FED_WORDS) {}

    void SiPixelRawToClusterGPUKernel::WordFedAppender::initializeWordFed(int fedId,
                                                                          unsigned int wordCounterGPU,
                                                                          const uint32_t *src,
                                                                          unsigned int length) {
      std::memcpy(word_.data() + wordCounterGPU, src, sizeof(uint32_t) * length);
      std::memset(fedId_.data() + wordCounterGPU / 2, fedId - 1200, length / 2);
    }

    ////////////////////
    KOKKOS_INLINE_FUNCTION uint32_t getLink(uint32_t ww) {
      return ((ww >> ::pixelgpudetails::LINK_shift) & ::pixelgpudetails::LINK_mask);
    }

    KOKKOS_INLINE_FUNCTION uint32_t getRoc(uint32_t ww) {
      return ((ww >> ::pixelgpudetails::ROC_shift) & ::pixelgpudetails::ROC_mask);
    }

    KOKKOS_INLINE_FUNCTION uint32_t getADC(uint32_t ww) {
      return ((ww >> ::pixelgpudetails::ADC_shift) & ::pixelgpudetails::ADC_mask);
    }

    KOKKOS_INLINE_FUNCTION bool isBarrel(uint32_t rawId) { return (1 == ((rawId >> 25) & 0x7)); }

    KOKKOS_INLINE_FUNCTION ::pixelgpudetails::DetIdGPU getRawId(const SiPixelFedCablingMapGPU *cablingMap,
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
    // Convert local pixel to pixelgpudetails::global pixel
    KOKKOS_INLINE_FUNCTION ::pixelgpudetails::Pixel frameConversion(
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

    KOKKOS_INLINE_FUNCTION uint8_t conversionError(uint8_t fedId, uint8_t status, bool debug = false) {
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

    KOKKOS_INLINE_FUNCTION bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol) {
      uint32_t numRowsInRoc = 80;
      uint32_t numColsInRoc = 52;

      /// row and collumn in ROC representation
      return ((rocRow < numRowsInRoc) & (rocCol < numColsInRoc));
    }

    KOKKOS_INLINE_FUNCTION bool dcolIsValid(uint32_t dcol, uint32_t pxid) {
      return ((dcol < 26) & (2 <= pxid) & (pxid < 162));
    }

    KOKKOS_INLINE_FUNCTION uint8_t checkROC(uint32_t errorWord,
                                            uint8_t fedId,
                                            uint32_t link,
                                            const SiPixelFedCablingMapGPU *cablingMap,
                                            bool debug = false) {
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

    KOKKOS_INLINE_FUNCTION uint32_t getErrRawID(uint8_t fedId,
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
    KOKKOS_FUNCTION void RawToDigi_kernel(const Kokkos::View<const SiPixelFedCablingMapGPU, KokkosExecSpace> cablingMap,
                                          const Kokkos::View<const unsigned char *, KokkosExecSpace> modToUnp,
                                          const uint32_t wordCounter,
                                          const Kokkos::View<unsigned int const *, KokkosExecSpace> word,
                                          const Kokkos::View<uint8_t const *, KokkosExecSpace> fedIds,
                                          Kokkos::View<uint16_t *, KokkosExecSpace> xx,
                                          Kokkos::View<uint16_t *, KokkosExecSpace> yy,
                                          Kokkos::View<uint16_t *, KokkosExecSpace> adc,
                                          Kokkos::View<uint32_t *, KokkosExecSpace> pdigi,
                                          Kokkos::View<uint32_t *, KokkosExecSpace> rawIdArr,
                                          Kokkos::View<uint16_t *, KokkosExecSpace> moduleId,
                                          cms::kokkos::SimpleVector<PixelErrorCompact> *err,
                                          const bool useQualityInfo,
                                          const bool includeErrors,
                                          const bool debug,
                                          const size_t gIndex) {
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
        return;
      }

      uint32_t link = getLink(ww);  // Extract link
      uint32_t roc = getRoc(ww);    // Extract Roc in link
      ::pixelgpudetails::DetIdGPU detId = getRawId(cablingMap.data(), fedId, link, roc);

      uint8_t errorType = checkROC(ww, fedId, link, cablingMap.data(), debug);
      skipROC = (roc < ::pixelgpudetails::maxROCIndex) ? false : (errorType != 0);
      if (includeErrors and skipROC) {
        uint32_t rID = getErrRawID(fedId, ww, errorType, cablingMap.data(), debug);
        err->push_back(PixelErrorCompact{rID, ww, errorType, fedId});
        return;
      }
      uint32_t rawId = detId.RawId;
      uint32_t rocIdInDetUnit = detId.rocInDet;
      bool barrel = isBarrel(rawId);

      uint32_t index = fedId * ::pixelgpudetails::MAX_LINK * ::pixelgpudetails::MAX_ROC +
                       (link - 1) * ::pixelgpudetails::MAX_ROC + roc;
      if (useQualityInfo) {
        skipROC = cablingMap().badRocs[index];
        if (skipROC)
          return;
      }
      skipROC = modToUnp[index];
      if (skipROC)
        return;

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
            err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
            if (debug)
              printf("BPIX1  Error status: %i\n", error);
            return;
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
          err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
          if (debug)
            printf("Error status: %i %d %d %d %d\n", error, dcol, pxid, fedId, roc);
          return;
        }
      }

      ::pixelgpudetails::Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
      xx[gIndex] = globalPix.row;  // origin shifting by 1 0-159
      yy[gIndex] = globalPix.col;  // origin shifting by 1 0-415
      adc[gIndex] = getADC(ww);
      pdigi[gIndex] = ::pixelgpudetails::pack(globalPix.row, globalPix.col, adc[gIndex]);
      moduleId[gIndex] = detId.moduleId;
      rawIdArr[gIndex] = rawId;
    }  // end of Raw to Digi kernel
  }    // namespace pixelgpudetails
}  // namespace KOKKOS_NAMESPACE

namespace pixelgpudetails {
  template <typename ExecSpace>
  void fillHitsModuleStart(Kokkos::View<uint32_t const *, ExecSpace> cluStart,
                           Kokkos::View<uint32_t *, ExecSpace> moduleStart,
                           ExecSpace const &execSpace) {
    assert(gpuClustering::MaxNumModules < 2048);  // easy to extend at least till 32*1024

    Kokkos::parallel_for(
        "fillHitsModuleStart_set_moduleStart",
        hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, gpuClustering::MaxNumModules)),
        KOKKOS_LAMBDA(const int &index) {
          moduleStart(index + 1) = std::min(gpuClustering::maxHitsInModule(), cluStart(index));
        });

    // limit to MaxHitsInModule;
    // for (int i = first, iend = gpuClustering::MaxNumModules; i < iend; i += blockDim.x) {
    //   moduleStart[i + 1] = std::min(gpuClustering::maxHitsInModule(), cluStart[i]);
    // }

    // __shared__ uint32_t ws[32];
    // blockPrefixScan(moduleStart + 1, moduleStart + 1, 1024, ws);
    Kokkos::parallel_scan(
        "fillHitsModuleStart_scanA",
        hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 1, 1025)),
        KOKKOS_LAMBDA(const int &i, float &upd, const bool &final) {
          upd += moduleStart[i];
          if (final)
            moduleStart[i] = upd;
        });
    // blockPrefixScan(moduleStart + 1025, moduleStart + 1025, gpuClustering::MaxNumModules - 1024, ws);
    Kokkos::parallel_scan(
        "fillHitsModuleStart_scanB",
        hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 1025, 1025 + gpuClustering::MaxNumModules - 1024)),
        KOKKOS_LAMBDA(const int &i, float &upd, const bool &final) {
          upd += moduleStart[i];
          if (final)
            moduleStart[i] = upd;
        });

    Kokkos::parallel_for(
        "fillHitsModuleStart_update_moduleStart",
        hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 1025, gpuClustering::MaxNumModules + 1)),
        KOKKOS_LAMBDA(const int &index) { moduleStart(index) += moduleStart(1024); });

#ifdef GPU_DEBUG
    Kokkos::parallel_for(
        "fillHitsModuleStart_debugA",
        hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, 1)),
        KOKKOS_LAMBDA(const int &index) {
          assert(0 == moduleStart(0));
          auto c0 = std::min(gpuClustering::maxHitsInModule(), cluStart(0));
          assert(c0 == moduleStart(1));
          assert(moduleStart(1024) >= moduleStart(1023));
          assert(moduleStart(1025) >= moduleStart(1024));
          assert(moduleStart(gpuClustering::MaxNumModules) >= moduleStart(1025));
        });

    Kokkos::parallel_for(
        "fillHitsModuleStart_debugB",
        hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, gpuClustering::MaxNumModules + 1)),
        KOKKOS_LAMBDA(const int &index) {
          if (0 != index)
            assert(moduleStart(i) >= moduleStart(i - i));
          // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
          // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
          if (index == 96 || index == 1184 || index == 1744 || index == (gpuClustering::MaxNumModules))
            printf("moduleStart %d %d\n", index, moduleStart(index));
        });

#endif

    // avoid overflow
    Kokkos::parallel_for(
        "fillHitsModuleStart_debugB",
        hintLightWeight(Kokkos::RangePolicy<ExecSpace>(execSpace, 0, gpuClustering::MaxNumModules + 1)),
        KOKKOS_LAMBDA(const int &index) {
          constexpr auto MAX_HITS = gpuClustering::MaxNumClusters;
          if (moduleStart(index) > MAX_HITS)
            moduleStart(index) = MAX_HITS;
        });
  }
}  // namespace pixelgpudetails

namespace KOKKOS_NAMESPACE {
  namespace pixelgpudetails {
    // Interface to outside
    void SiPixelRawToClusterGPUKernel::makeClustersAsync(
        bool isRun2,
        const Kokkos::View<const SiPixelFedCablingMapGPU, KokkosExecSpace> &cablingMap,
        const Kokkos::View<const unsigned char *, KokkosExecSpace> &modToUnp,
        const SiPixelGainForHLTonGPU<KokkosExecSpace> &gains,
        const WordFedAppender &wordFed,
        PixelFormatterErrors &&errors,
        const uint32_t wordCounter,
        const uint32_t fedCounter,
        bool useQualityInfo,
        bool includeErrors,
        bool debug,
        KokkosExecSpace const &execSpace) {
      nDigis = wordCounter;

#ifdef GPU_DEBUG
      std::cout << "decoding " << wordCounter << " digis. Max is " << pixelgpudetails::MAX_FED_WORDS << std::endl;
#endif

      digis_d = SiPixelDigisKokkos<KokkosExecSpace>(pixelgpudetails::MAX_FED_WORDS);
      if (includeErrors) {
        digiErrors_d = SiPixelDigiErrorsKokkos<KokkosExecSpace>(
            pixelgpudetails::MAX_FED_WORDS, std::move(errors), KokkosExecSpace());
      }
      clusters_d = SiPixelClustersKokkos<KokkosExecSpace>(::gpuClustering::MaxNumModules);

      if (wordCounter)  // protect in case of empty event....
      {
        assert(0 == wordCounter % 2);
        // wordCounter is the total no of words in each event to be trasfered on device

        // TODO: can not deep_copy Views of different size
        //Kokkos::View<unsigned int *, KokkosExecSpace> word_d("word_d", wordCounter);
        //Kokkos::View<unsigned char *, KokkosExecSpace> fedId_d("fedId_d", wordCounter);
        Kokkos::View<unsigned int *, KokkosExecSpace> word_d(Kokkos::ViewAllocateWithoutInitializing("word_d"),
                                                             MAX_FED_WORDS);
        Kokkos::View<unsigned char *, KokkosExecSpace> fedId_d(Kokkos::ViewAllocateWithoutInitializing("fedId_d"),
                                                               MAX_FED_WORDS);
        Kokkos::deep_copy(execSpace, word_d, wordFed.word());
        Kokkos::deep_copy(execSpace, fedId_d, wordFed.fedId());

        {
          // need Kokkos::Views as local variables to pass to the lambda
          auto xx_d = digis_d.xx();
          auto yy_d = digis_d.yy();
          auto adc_d = digis_d.adc();
          auto pdigi_d = digis_d.pdigi();
          auto rawIdArr_d = digis_d.rawIdArr();
          auto moduleInd_d = digis_d.moduleInd();
          auto error_d = digiErrors_d.error();  // returns nullptr if default-constructed

          Kokkos::parallel_for(
              "RawToDigi_kernel",
              hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, wordCounter)),
              KOKKOS_LAMBDA(const size_t i) {
                RawToDigi_kernel(cablingMap,
                                 modToUnp,
                                 wordCounter,
                                 word_d,
                                 fedId_d,
                                 xx_d,
                                 yy_d,
                                 adc_d,
                                 pdigi_d,
                                 rawIdArr_d,
                                 moduleInd_d,
                                 error_d,
                                 useQualityInfo,
                                 includeErrors,
                                 debug,
                                 i);
              });
        }
#ifdef TODO
        if (includeErrors) {
          digiErrors_d.copyErrorToHostAsync(stream);
        }
#endif
      }
      // End of Raw2Digi and passing data for clustering

      {
        // clusterizer ...
        {
          auto xx_d = digis_d.c_xx();
          auto yy_d = digis_d.c_yy();
          auto adc_d = digis_d.adc();
          auto moduleInd_d = digis_d.moduleInd();
          auto moduleStart_d = clusters_d.moduleStart();
          auto clusInModule_d = clusters_d.clusInModule();
          auto clusModuleStart_d = clusters_d.clusModuleStart();

          Kokkos::parallel_for(
              "calibDigis",
              hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(
                  execSpace, 0, std::max(int(wordCounter), int(::gpuClustering::MaxNumModules)))),
              KOKKOS_LAMBDA(const size_t i) {
                gpuCalibPixel::calibDigis(isRun2,
                                          moduleInd_d,
                                          xx_d,
                                          yy_d,
                                          adc_d,
                                          gains,
                                          wordCounter,
                                          moduleStart_d,
                                          clusInModule_d,
                                          clusModuleStart_d,
                                          i);
              });
        }

#ifdef GPU_DEBUG
        execSpace.fence();
#endif

#ifdef GPU_DEBUG
        std::cout << "CUDA countModules kernel launch with " << blocks << " blocks of " << threadsPerBlock
                  << " threads\n";
#endif

        {
          auto moduleInd_d = digis_d.moduleInd();
          auto moduleStart_d = clusters_d.moduleStart();
          auto clusStart_d = digis_d.clus();
          Kokkos::parallel_for(
              "countModules",
              hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(
                  execSpace, 0, std::max(int(wordCounter), int(::gpuClustering::MaxNumModules)))),
              KOKKOS_LAMBDA(const size_t i) {
                gpuClustering::countModules(moduleInd_d, moduleStart_d, clusStart_d, wordCounter, i);
              });
        }

        // read the number of modules into a data member, used by getProduct())
        Kokkos::deep_copy(
            execSpace, Kokkos::subview(nModules_Clusters_h, 0), Kokkos::subview(clusters_d.moduleStart(), 0));

        const uint32_t blocks = ::gpuClustering::MaxNumModules;
        Kokkos::TeamPolicy<KokkosExecSpace> teamPolicy(execSpace, blocks, Kokkos::AUTO());
#ifdef GPU_DEBUG
        std::cout << "CUDA findClus kernel launch with " << blocks << " blocks of " << teamPolicy.team_size()
                  << " threads\n";
#endif

        ::gpuClustering::findClus<KokkosExecSpace>(digis_d.c_moduleInd(),
                                                   digis_d.c_xx(),
                                                   digis_d.c_yy(),
                                                   clusters_d.c_moduleStart(),
                                                   clusters_d.clusInModule(),
                                                   clusters_d.moduleId(),
                                                   digis_d.clus(),
                                                   int(wordCounter),
                                                   teamPolicy,
                                                   execSpace);

#ifdef GPU_DEBUG
        execSpace.fence();
#endif

        // apply charge cut
        ::gpuClustering::clusterChargeCut<KokkosExecSpace>(digis_d.moduleInd(),
                                                           digis_d.c_adc(),
                                                           clusters_d.c_moduleStart(),
                                                           clusters_d.clusInModule(),
                                                           clusters_d.c_moduleId(),
                                                           digis_d.clus(),
                                                           int(wordCounter),
                                                           teamPolicy,
                                                           execSpace);

        // count the module start indices already here (instead of
        // rechits) so that the number of clusters/hits can be made
        // available in the rechit producer without additional points of
        // synchronization/ExternalWork

        ::pixelgpudetails::fillHitsModuleStart(clusters_d.c_clusInModule(), clusters_d.clusModuleStart(), execSpace);

        // last element holds the number of all clusters
        Kokkos::deep_copy(execSpace,
                          Kokkos::subview(nModules_Clusters_h, 1),
                          Kokkos::subview(clusters_d.clusModuleStart(), ::gpuClustering::MaxNumModules));

#ifdef GPU_DEBUG
        execSpace.fence();
#endif
      }  // end clusterizer scope
    }
  }  // namespace pixelgpudetails
}  // namespace KOKKOS_NAMESPACE
