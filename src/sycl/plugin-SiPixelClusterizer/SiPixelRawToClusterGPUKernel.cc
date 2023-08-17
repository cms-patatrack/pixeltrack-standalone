/* Sushil Dubey, Shashi Dugad, TIFR, July 2017
 *
 * File Name: RawToClusterGPU.cu
 * Description: It converts Raw data into Digi Format on GPU
 * Finaly the Output of RawToDigi data is given to pixelClusterizer
 *
**/

// C++ includes
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// SYCL includes
#include <sycl/sycl.hpp>

// CMSSW includes
#include "SYCLDataFormats/gpuClusteringConstants.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "SYCLCore/sycl_assert.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"

// local includes
#include "SiPixelRawToClusterGPUKernel.h"
#include "gpuCalibPixel.h"
#include "gpuClusterChargeCut.h"
#include "gpuClustering.h"

namespace pixelgpudetails {

  // number of words for all the FEDs
  constexpr uint32_t MAX_FED_WORDS = pixelgpudetails::MAX_FED * pixelgpudetails::MAX_WORD;

  SiPixelRawToClusterGPUKernel::WordFedAppender::WordFedAppender()
      : word_(new unsigned int[MAX_FED_WORDS]), fedId_(new unsigned char[MAX_FED_WORDS]) {}

  void SiPixelRawToClusterGPUKernel::WordFedAppender::initializeWordFed(int fedId,
                                                                        unsigned int wordCounterGPU,
                                                                        const uint32_t *src,
                                                                        unsigned int length) {
    std::memcpy(word_.get() + wordCounterGPU, src, sizeof(uint32_t) * length);
    std::memset(fedId_.get() + wordCounterGPU / 2, fedId - 1200, length / 2);
  }

  ////////////////////

  uint32_t getLink(uint32_t ww) { return ((ww >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask); }

  uint32_t getRoc(uint32_t ww) { return ((ww >> pixelgpudetails::ROC_shift) & pixelgpudetails::ROC_mask); }

  uint32_t getADC(uint32_t ww) { return ((ww >> pixelgpudetails::ADC_shift) & pixelgpudetails::ADC_mask); }

  bool isBarrel(uint32_t rawId) { return (1 == ((rawId >> 25) & 0x7)); }

  pixelgpudetails::DetIdGPU getRawId(const SiPixelFedCablingMapGPU *cablingMap,
                                     uint8_t fed,
                                     uint32_t link,
                                     uint32_t roc) {
    uint32_t index = fed * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
    pixelgpudetails::DetIdGPU detId = {
        cablingMap->RawId[index], cablingMap->rocInDet[index], cablingMap->moduleId[index]};
    return detId;
  }

  //reference http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/dd/d31/FrameConversion_8cc_source.html
  //http://cmslxr.fnal.gov/source/CondFormats/SiPixelObjects/src/PixelROC.cc?v=CMSSW_9_2_0#0071
  // Convert local pixel to pixelgpudetails::global pixel
  pixelgpudetails::Pixel frameConversion(
      bool bpix, int side, uint32_t layer, uint32_t rocIdInDetUnit, pixelgpudetails::Pixel local) {
    int slopeRow = 0, slopeCol = 0;
    int rowOffset = 0, colOffset = 0;

    if (bpix) {
      if (side == -1 && layer != 1) {  // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }       // if roc
      } else {  // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
        if (rocIdInDetUnit < 8) {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = rocIdInDetUnit * pixelgpudetails::numColsInRoc;
        } else {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (16 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        }
      }

    } else {             // fpix
      if (side == -1) {  // pannel 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }
      } else {  // pannel 2
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }

      }  // side
    }

    uint32_t gRow = rowOffset + slopeRow * local.row;
    uint32_t gCol = colOffset + slopeCol * local.col;
    pixelgpudetails::Pixel global = {gRow, gCol};
    return global;
  }

  uint8_t conversionError(uint8_t fedId, uint8_t status, bool debug = false) {
    uint8_t errorType = 0;

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

  bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol) {
    uint32_t numRowsInRoc = 80;
    uint32_t numColsInRoc = 52;

    /// row and collumn in ROC representation
    return ((rocRow < numRowsInRoc) & (rocCol < numColsInRoc));
  }

  bool dcolIsValid(uint32_t dcol, uint32_t pxid) { return ((dcol < 26) & (2 <= pxid) & (pxid < 162)); }

  uint8_t checkROC(
      uint32_t errorWord, uint8_t fedId, uint32_t link, const SiPixelFedCablingMapGPU *cablingMap, bool debug = false) {
    uint8_t errorType = (errorWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ERROR_mask;
    if (errorType >= 25)
      printf("errortype is : %d\n", errorType);
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
        if ((errorWord >> pixelgpudetails::OMIT_ERR_shift) & pixelgpudetails::OMIT_ERR_mask) {
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

    return 0;
  }

  uint32_t getErrRawID(uint8_t fedId,
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
        uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
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
        uint32_t roc = (errWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ROC_mask;
        uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
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
  void RawToDigi_kernel(const SiPixelFedCablingMapGPU *cablingMap,
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
                        cms::sycltools::SimpleVector<PixelErrorCompact> *err,
                        bool useQualityInfo,
                        bool includeErrors,
                        bool debug,
                        sycl::nd_item<1> item) {
    int32_t first = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
    for (int32_t iloop = first, nend = wordCounter; iloop < nend;
         iloop += item.get_local_range(0) * item.get_group_range(0)) {
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
      pixelgpudetails::DetIdGPU detId = getRawId(cablingMap, fedId, link, roc);

      uint8_t errorType = checkROC(ww, fedId, link, cablingMap, debug);

      skipROC = (roc < pixelgpudetails::maxROCIndex) ? false : (errorType != 0);
      if (includeErrors and skipROC) {
        uint32_t rID = getErrRawID(fedId, ww, errorType, cablingMap, debug);
        err->push_back(PixelErrorCompact{rID, ww, errorType, fedId});
        continue;
      }

      uint32_t rawId = detId.RawId;
      uint32_t rocIdInDetUnit = detId.rocInDet;
      bool barrel = isBarrel(rawId);

      uint32_t index = fedId * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
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
        layer = (rawId >> pixelgpudetails::layerStartBit) & pixelgpudetails::layerMask;
        module = (rawId >> pixelgpudetails::moduleStartBit) & pixelgpudetails::moduleMask;
        side = (module < 5) ? -1 : 1;
      } else {
        // endcap ids
        layer = 0;
        panel = (rawId >> pixelgpudetails::panelStartBit) & pixelgpudetails::panelMask;
        //disk  = (rawId >> diskStartBit_) & diskMask_;
        side = (panel == 1) ? -1 : 1;
        //blade = (rawId >> bladeStartBit_) & bladeMask_;
      }

      // ***special case of layer to 1 be handled here
      pixelgpudetails::Pixel localPix;
      if (layer == 1) {
        uint32_t col = (ww >> pixelgpudetails::COL_shift) & pixelgpudetails::COL_mask;
        uint32_t row = (ww >> pixelgpudetails::ROW_shift) & pixelgpudetails::ROW_mask;
        localPix.row = row;
        localPix.col = col;
        if (includeErrors) {
          if (not rocRowColIsValid(row, col)) {
            uint8_t error = conversionError(fedId, 3, debug);  //use the device function and fill the arrays
            err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
            if (debug)
              printf("BPIX1  Error status: %i\n", error);
            continue;
          }
        }
      } else {
        // ***conversion rules for dcol and pxid
        uint32_t dcol = (ww >> pixelgpudetails::DCOL_shift) & pixelgpudetails::DCOL_mask;
        uint32_t pxid = (ww >> pixelgpudetails::PXID_shift) & pixelgpudetails::PXID_mask;
        uint32_t row = pixelgpudetails::numRowsInRoc - pxid / 2;
        uint32_t col = dcol * 2 + pxid % 2;
        localPix.row = row;
        localPix.col = col;
        if (includeErrors and not dcolIsValid(dcol, pxid)) {
          uint8_t error = conversionError(fedId, 3, debug);
          err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
          if (debug)
            printf("Error status: %i %d %d %d %d\n", error, dcol, pxid, fedId, roc);
          continue;
        }
      }

      pixelgpudetails::Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
      xx[gIndex] = globalPix.row;  // origin shifting by 1 0-159
      yy[gIndex] = globalPix.col;  // origin shifting by 1 0-415
      adc[gIndex] = getADC(ww);
      pdigi[gIndex] = pixelgpudetails::pack(globalPix.row, globalPix.col, adc[gIndex]);
      moduleId[gIndex] = detId.moduleId;
      rawIdArr[gIndex] = rawId;
    }  // end of loop (gIndex < end)
  }    // end of Raw to Digi kernel

  void fillHitsModuleStart(uint32_t const *__restrict__ cluStart,
                           uint32_t *__restrict__ moduleStart,
                           sycl::nd_item<1> item) {
    assert(gpuClustering::MaxNumModules < 2048);  // easy to extend at least till 32*1024
    assert(1 == item.get_group_range(0));
    assert(0 == item.get_group(0));

    int first = item.get_local_id(0);

    // limit to MaxHitsInModule;
    for (int i = first, iend = gpuClustering::MaxNumModules; i < iend; i += item.get_local_range(0)) {
      moduleStart[i + 1] = std::min(gpuClustering::maxHitsInModule(), cluStart[i]);
    }

    auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint16_t[32]>(item.get_group());
    uint16_t *ws = (uint16_t *)wsbuff.get();

    cms::sycltools::blockPrefixScan(moduleStart + 1, moduleStart + 1, 1024, item, ws);
    cms::sycltools::blockPrefixScan(
        moduleStart + 1025, moduleStart + 1025, gpuClustering::MaxNumModules - 1024, item, ws);

    for (int i = first + 1025, iend = gpuClustering::MaxNumModules + 1; i < iend; i += item.get_local_range(0)) {
      moduleStart[i] += moduleStart[1024];
    }

    sycl::group_barrier(item.get_group());

#ifdef GPU_DEBUG
    assert(0 == moduleStart[0]);
    auto c0 = std::min(gpuClustering::maxHitsInModule(), cluStart[0]);
    assert(c0 == moduleStart[1]);
    assert(moduleStart[1024] >= moduleStart[1023]);
    assert(moduleStart[1025] >= moduleStart[1024]);
    assert(moduleStart[gpuClustering::MaxNumModules] >= moduleStart[1025]);

    for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += item.get_local_range(0)) {
      if (0 != i)
        assert(moduleStart[i] >= moduleStart[i - i]);
      // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
      // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
      if (i == 96 || i == 1184 || i == 1744 || i == gpuClustering::MaxNumModules)
        printf("moduleStart [%d] = %d\n", i, moduleStart[i]);
    }
#endif

    // avoid overflow
    constexpr auto MAX_HITS = gpuClustering::MaxNumClusters;
    for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += item.get_local_range(0)) {
      if (moduleStart[i] > MAX_HITS)
        moduleStart[i] = MAX_HITS;
    }
  }

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
                                                       sycl::queue stream,
                                                       bool isCpu) {
    nDigis = wordCounter;
#ifdef GPU_DEBUG
    std::cout << "decoding " << wordCounter << " digis. Max is " << pixelgpudetails::MAX_FED_WORDS << std::endl;
#endif

    digis_d = SiPixelDigisSYCL(pixelgpudetails::MAX_FED_WORDS, stream);
    if (includeErrors) {
      digiErrors_d = SiPixelDigiErrorsSYCL(pixelgpudetails::MAX_FED_WORDS, std::move(errors), stream);
    }
    clusters_d = SiPixelClustersSYCL(gpuClustering::MaxNumModules, stream);
    nModules_Clusters_h = cms::sycltools::make_host_unique<uint32_t[]>(2, stream);
    if (wordCounter)  // protect in case of empty event....
    {
      const int threadsPerBlock = 256;
      const int blocks = (wordCounter + threadsPerBlock - 1) / threadsPerBlock;  // fill it all
      sycl::range<1> numthreadsPerBlock(threadsPerBlock);
      sycl::range<1> globalSize(blocks * threadsPerBlock);
      assert(0 == wordCounter % 2);

      auto word_d = cms::sycltools::make_device_unique<uint32_t[]>(wordCounter, stream);
      auto fedId_d = cms::sycltools::make_device_unique<uint8_t[]>(wordCounter, stream);

      stream.memcpy(word_d.get(), wordFed.word(), sizeof(uint32_t) * wordCounter);
      stream.memcpy(fedId_d.get(), wordFed.fedId(), sizeof(uint8_t) * wordCounter / 2);

      stream.submit([&](sycl::handler &cgh) {
        auto cablingMap_kernel = cablingMap;
        auto modToUnp_kernel = modToUnp;
        auto word_d_kernel = word_d.get();
        auto fedId_d_kernel = fedId_d.get();
        auto digis_x_kernel = digis_d.xx();
        auto digis_y_kernel = digis_d.yy();
        auto digis_adc_kernel = digis_d.adc();
        auto digis_digi_kernel = digis_d.pdigi();
        auto digis_raw_kernel = digis_d.rawIdArr();
        auto digis_mod_kernel = digis_d.moduleInd();
        auto digiErrors_d_kernel = digiErrors_d.error();
        cgh.parallel_for<class rawToDigi_Kernel>(
            sycl::nd_range<1>(globalSize, numthreadsPerBlock), [=](sycl::nd_item<1> item) {
              RawToDigi_kernel(cablingMap_kernel,
                               modToUnp_kernel,
                               wordCounter,
                               word_d_kernel,
                               fedId_d_kernel,
                               digis_x_kernel,
                               digis_y_kernel,
                               digis_adc_kernel,
                               digis_digi_kernel,
                               digis_raw_kernel,
                               digis_mod_kernel,
                               digiErrors_d_kernel,  // returns nullptr if default-constructed
                               useQualityInfo,
                               includeErrors,
                               debug,
                               item);
            });
      });

#ifdef GPU_DEBUG
      stream.wait();
#endif
    }
    // End of Raw2Digi and passing data for clustering
    {
      // clusterizer ...
      using namespace gpuClustering;
      int threadsPerBlock = 256;
      int blocks =
          (std::max(int(wordCounter), int(gpuClustering::MaxNumModules)) + threadsPerBlock - 1) / threadsPerBlock;
      stream.submit([&](sycl::handler &cgh) {
        auto digis_x_kernel = digis_d.c_xx();
        auto digis_y_kernel = digis_d.c_yy();
        auto digis_adc_kernel = digis_d.adc();
        auto digis_ind_kernel = digis_d.moduleInd();
        auto gains_kernel = gains;
        auto clusters_d_kernel = clusters_d.moduleStart();
        auto clusters_in_kernel = clusters_d.clusInModule();
        auto clusters_cs_kernel = clusters_d.clusModuleStart();
        cgh.parallel_for<class calibDigis_kernel>(
            sycl::nd_range<1>(sycl::range<1>(blocks) * sycl::range<1>(threadsPerBlock),
                              sycl::range<1>(threadsPerBlock)),
            [=](sycl::nd_item<1> item) {
              gpuCalibPixel::calibDigis(isRun2,
                                        digis_ind_kernel,
                                        digis_x_kernel,
                                        digis_y_kernel,
                                        digis_adc_kernel,
                                        gains_kernel,
                                        wordCounter,
                                        clusters_d_kernel,
                                        clusters_in_kernel,
                                        clusters_cs_kernel,
                                        item);
            });
      });
#ifdef GPU_DEBUG
      stream.wait();
#endif

#ifdef GPU_DEBUG
      std::cout << "SYCL countModules kernel launch with " << blocks << " blocks of " << threadsPerBlock
                << " threads\n";
#endif
      stream.submit([&](sycl::handler &cgh) {
        auto digis_ind_kernel = digis_d.c_moduleInd();
        auto clusters_d_kernel = clusters_d.moduleStart();
        auto digis_d_kernel = digis_d.clus();
        cgh.parallel_for<class countModules_kernel>(
            sycl::nd_range<1>(sycl::range<1>(blocks) * sycl::range<1>(threadsPerBlock),
                              sycl::range<1>(threadsPerBlock)),
            [=](sycl::nd_item<1> item) {
              countModules(digis_ind_kernel, clusters_d_kernel, digis_d_kernel, wordCounter, item);
            });
      });

      // read the number of modules into a data member, used by getProduct())
      stream.memcpy(&(nModules_Clusters_h[0]), clusters_d.moduleStart(), sizeof(uint32_t));

      threadsPerBlock = 256;  //SYCL_BUG_ 256 for GPU, set to 32 (and change values in the kernel for CPU)
      blocks = MaxNumModules;
#ifdef GPU_DEBUG
      std::cout << "SYCL findClus kernel launch with " << blocks << " blocks of " << threadsPerBlock << " threads\n";
#endif
      constexpr uint32_t maxPixInModule = 4000;
      constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;
      using Hist = cms::sycltools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
      if (isCpu) {
        threadsPerBlock = 32;
        stream.submit([&](sycl::handler &cgh) {
          auto digis_x_kernel = digis_d.c_xx();
          auto digis_y_kernel = digis_d.c_yy();
          auto digis_ind_kernel = digis_d.c_moduleInd();
          auto digis_clus_kernel = digis_d.clus();
          auto clusters_s_kernel = clusters_d.c_moduleStart();
          auto clusters_in_kernel = clusters_d.clusInModule();
          auto clusters_id_kernel = clusters_d.moduleId();
          cgh.parallel_for<class findClusCPU_kernel>(
              sycl::nd_range<1>(sycl::range<1>(blocks * threadsPerBlock), sycl::range<1>(threadsPerBlock)),
              [=](sycl::nd_item<1> item)
                  [[intel::reqd_sub_group_size(32)]] {  // explicitly specify sub-group size (32 is the maximum)
                    findClusCPU(digis_ind_kernel,
                                digis_x_kernel,
                                digis_y_kernel,
                                clusters_s_kernel,
                                clusters_in_kernel,
                                clusters_id_kernel,
                                digis_clus_kernel,
                                wordCounter,
                                item);
                  });
        });
      } else {
        stream.submit([&](sycl::handler &cgh) {
          auto digis_x_kernel = digis_d.c_xx();
          auto digis_y_kernel = digis_d.c_yy();
          auto digis_ind_kernel = digis_d.c_moduleInd();
          auto digis_clus_kernel = digis_d.clus();
          auto clusters_s_kernel = clusters_d.c_moduleStart();
          auto clusters_in_kernel = clusters_d.clusInModule();
          auto clusters_id_kernel = clusters_d.moduleId();
          cgh.parallel_for<class findClusGPU_kernel>(
              sycl::nd_range<1>(sycl::range<1>(blocks * threadsPerBlock), sycl::range<1>(threadsPerBlock)),
              [=](sycl::nd_item<1> item)
                  [[intel::reqd_sub_group_size(32)]] {  // explicitly specify sub-group size (32 is the maximum)
                    findClusGPU(digis_ind_kernel,
                                digis_x_kernel,
                                digis_y_kernel,
                                clusters_s_kernel,
                                clusters_in_kernel,
                                clusters_id_kernel,
                                digis_clus_kernel,
                                wordCounter,
                                item);
                  });
        });
      }
#ifdef GPU_DEBUG
      stream.wait();
#endif
      threadsPerBlock = 256;
      blocks = MaxNumModules;

      stream.submit([&](sycl::handler &cgh) {
        // apply charge cut
        auto digis_ind_kernel = digis_d.moduleInd();
        auto digis_adc_kernel = digis_d.c_adc();
        auto clusters_s_kernel = clusters_d.c_moduleStart();
        auto clusters_in_kernel = clusters_d.clusInModule();
        auto clusters_cs_kernel = clusters_d.c_moduleId();
        auto digis_clus_kernel = digis_d.clus();
        cgh.parallel_for<class clusterChargeCut_kernel>(
            sycl::nd_range<1>(sycl::range<1>(blocks * threadsPerBlock), sycl::range<1>(threadsPerBlock)),
            [=](sycl::nd_item<1> item)
                [[intel::reqd_sub_group_size(32)]] {  // explicitly specify sub-group size (32 is the maximum)
                  clusterChargeCut(digis_ind_kernel,
                                   digis_adc_kernel,
                                   clusters_s_kernel,
                                   clusters_in_kernel,
                                   clusters_cs_kernel,
                                   digis_clus_kernel,
                                   wordCounter,
                                   item);
                });
      });

      // count the module start indices already here (instead of
      // rechits) so that the number of clusters/hits can be made
      // available in the rechit producer without additional points of
      // synchronization/ExternalWor
      stream.submit([&](sycl::handler &cgh) {
        // apply charge cut
        auto clusters_in_kernel = clusters_d.c_clusInModule();
        auto clusters_s_kernel = clusters_d.clusModuleStart();
        cgh.parallel_for<class fillHitsModuleStart_kernel>(
            sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)),
            [=](sycl::nd_item<1> item)
                [[intel::reqd_sub_group_size(32)]] {  // explicitly specify sub-group size (32 is the maximum)
                  fillHitsModuleStart(clusters_in_kernel, clusters_s_kernel, item);
                });
      });
      // MUST be ONE block

      // last element holds the number of all clusters
      stream.memcpy(
          &(nModules_Clusters_h[1]), clusters_d.clusModuleStart() + gpuClustering::MaxNumModules, sizeof(int32_t));
#ifdef GPU_DEBUG
      stream.wait();
      std::cout << "Number of modules (nModules_Clusters_h[0]): " << nModules_Clusters_h[0]
                << " and number of clusters (nModules_Clusters_h[1]): " << nModules_Clusters_h[1] << std::endl;
#endif
#ifdef __SYCL_TARGET_INTEL_X86_64__
      // FIXME needed only on CPU ?
      stream.wait();
#endif
    }  // end clusterizer scope
  }
}  // namespace pixelgpudetails
