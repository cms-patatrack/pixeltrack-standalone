#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h

#include <algorithm>
#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "CUDADataFormats/SiPixelDigiErrorsCUDA.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"
#include "CUDACore/SimpleVector.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/host_noncached_unique_ptr.h"
#include "DataFormats/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelFormatterErrors.h"

// local include(s)
#include "SiPixelClusterThresholds.h"

struct SiPixelROCsStatusAndMapping;
class SiPixelGainForHLTonGPU;

namespace pixelgpudetails {

  inline namespace phase1geometry {
    const uint32_t layerStartBit = 20;
    const uint32_t ladderStartBit = 12;
    const uint32_t moduleStartBit = 2;

    const uint32_t panelStartBit = 10;
    const uint32_t diskStartBit = 18;
    const uint32_t bladeStartBit = 12;

    const uint32_t layerMask = 0xF;
    const uint32_t ladderMask = 0xFF;
    const uint32_t moduleMask = 0x3FF;
    const uint32_t panelMask = 0x3;
    const uint32_t diskMask = 0xF;
    const uint32_t bladeMask = 0x3F;
  }  // namespace phase1geometry

  const uint32_t maxROCIndex = 8;
  const uint32_t numRowsInRoc = 80;
  const uint32_t numColsInRoc = 52;

  const uint32_t MAX_WORD = 2000;

  struct Pixel {
    uint32_t row;
    uint32_t col;
  };

  class Packing {
  public:
    using PackedDigiType = uint32_t;

    // Constructor: pre-computes masks and shifts from field widths
    __host__ __device__ inline constexpr Packing(unsigned int row_w,
                                                 unsigned int column_w,
                                                 unsigned int time_w,
                                                 unsigned int adc_w)
        : row_width(row_w),
          column_width(column_w),
          adc_width(adc_w),
          row_shift(0),
          column_shift(row_shift + row_w),
          time_shift(column_shift + column_w),
          adc_shift(time_shift + time_w),
          row_mask(~(~0U << row_w)),
          column_mask(~(~0U << column_w)),
          time_mask(~(~0U << time_w)),
          adc_mask(~(~0U << adc_w)),
          rowcol_mask(~(~0U << (column_w + row_w))),
          max_row(row_mask),
          max_column(column_mask),
          max_adc(adc_mask) {}

    uint32_t row_width;
    uint32_t column_width;
    uint32_t adc_width;

    uint32_t row_shift;
    uint32_t column_shift;
    uint32_t time_shift;
    uint32_t adc_shift;

    PackedDigiType row_mask;
    PackedDigiType column_mask;
    PackedDigiType time_mask;
    PackedDigiType adc_mask;
    PackedDigiType rowcol_mask;

    uint32_t max_row;
    uint32_t max_column;
    uint32_t max_adc;
  };

  __host__ __device__ inline constexpr Packing packing() { return Packing(11, 11, 0, 10); }

  __host__ __device__ inline uint32_t pack(uint32_t row, uint32_t col, uint32_t adc) {
    constexpr Packing thePacking = packing();
    adc = std::min(adc, thePacking.max_adc);

    return (row << thePacking.row_shift) | (col << thePacking.column_shift) | (adc << thePacking.adc_shift);
  }

  constexpr uint32_t pixelToChannel(int row, int col) {
    constexpr Packing thePacking = packing();
    return (row << thePacking.column_width) | col;
  }

  class SiPixelRawToClusterGPUKernel {
  public:
    class WordFedAppender {
    public:
      WordFedAppender();
      ~WordFedAppender() = default;

      void initializeWordFed(int fedId, unsigned int wordCounterGPU, const uint32_t* src, unsigned int length);

      const unsigned int* word() const { return word_.get(); }
      const unsigned char* fedId() const { return fedId_.get(); }

    private:
      cms::cuda::host::noncached::unique_ptr<unsigned int[]> word_;
      cms::cuda::host::noncached::unique_ptr<unsigned char[]> fedId_;
    };

    SiPixelRawToClusterGPUKernel() = default;
    ~SiPixelRawToClusterGPUKernel() = default;

    SiPixelRawToClusterGPUKernel(const SiPixelRawToClusterGPUKernel&) = delete;
    SiPixelRawToClusterGPUKernel(SiPixelRawToClusterGPUKernel&&) = delete;
    SiPixelRawToClusterGPUKernel& operator=(const SiPixelRawToClusterGPUKernel&) = delete;
    SiPixelRawToClusterGPUKernel& operator=(SiPixelRawToClusterGPUKernel&&) = delete;

    void makeClustersAsync(bool isRun2,
                           const SiPixelClusterThresholds clusterThresholds,
                           const SiPixelROCsStatusAndMapping &cablingMap,
                           const unsigned char* modToUnp,
                           const SiPixelGainForHLTonGPU* gains,
                           const WordFedAppender& wordFed,
                           SiPixelFormatterErrors&& errors,
                           const uint32_t wordCounter,
                           const uint32_t fedCounter,
                           bool useQualityInfo,
                           bool includeErrors,
                           bool debug,
                           cudaStream_t stream);

    std::pair<SiPixelDigisCUDA, SiPixelClustersCUDA> getResults() {
      digis_d.setNModulesDigis(nModules_Clusters_h[0], nDigis);
      clusters_d.setNClusters(nModules_Clusters_h[1]);
      // need to explicitly deallocate while the associated CUDA
      // stream is still alive
      //
      // technically the statement above is not true anymore now that
      // the CUDA streams are cached within the cms::cuda::StreamCache, but it is
      // still better to release as early as possible
      nModules_Clusters_h.reset();
      return std::make_pair(std::move(digis_d), std::move(clusters_d));
    }

    SiPixelDigiErrorsCUDA&& getErrors() { return std::move(digiErrors_d); }

  private:
    uint32_t nDigis = 0;

    // Data to be put in the event
    cms::cuda::host::unique_ptr<uint32_t[]> nModules_Clusters_h;
    SiPixelDigisCUDA digis_d;
    SiPixelClustersCUDA clusters_d;
    SiPixelDigiErrorsCUDA digiErrors_d;
  };

}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h
