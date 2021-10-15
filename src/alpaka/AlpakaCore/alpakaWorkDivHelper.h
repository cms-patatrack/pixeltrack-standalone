#ifndef AlpakaCore_alpakaWorkDivHelper_h
#define AlpakaCore_alpakaWorkDivHelper_h

#include "AlpakaCore/alpakaConfig.h"

using namespace alpaka_common;

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  /*********************************************
   *              WORKDIV CREATION
   ********************************************/

  /*
   * Creates the accelerator-dependent workdiv.
   */
  template <typename TDim>
  WorkDiv<TDim> make_workdiv(const Vec<TDim>& blocksPerGrid, const Vec<TDim>& threadsPerBlockOrElementsPerThread) {
    // On the GPU:
    // threadsPerBlockOrElementsPerThread is the number of threads per block.
    // Each thread is looking at a single element: elementsPerThread is always 1.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    const Vec<TDim>& elementsPerThread = Vec<TDim>::ones();
    return WorkDiv<TDim>(blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
#else
    // On the CPU:
    // Run serially with a single thread per block: threadsPerBlock is always 1.
    // threadsPerBlockOrElementsPerThread is the number of elements per thread.
    const Vec<TDim>& threadsPerBlock = Vec<TDim>::ones();
    return WorkDiv<TDim>(blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
#endif
  }

  /*********************************************
   *           RANGE COMPUTATION
   ********************************************/

  /*
   * Computes the range of the elements indexes, local to the block.
   * Warning: the max index is not truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block(const TAcc& acc,
                                                                 const Idx elementIdxShift,
                                                                 const unsigned int dimIndex = 0u) {
    // Take into account the thread index in block.
    const Idx threadIndex(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndex]);
    const Idx threadDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[dimIndex]);

    // Compute the elements indexes in block.
    // Obviously relevant for CPU only.
    // For GPU, threadDimension == 1, and elementIdx == firstElementIdx == threadIdx + elementIdxShift.
    const Idx firstElementIdxLocal = threadIndex * threadDimension;
    const Idx firstElementIdx = firstElementIdxLocal + elementIdxShift;  // Add the shift!
    const Idx endElementIdxUncut = firstElementIdx + threadDimension;

    // Return element indexes, shifted by elementIdxShift.
    return {firstElementIdx, endElementIdxUncut};
  }

  /*
   * Computes the range of the elements indexes, local to the block.
   * Truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block_truncated(const TAcc& acc,
                                                                           const Idx maxNumberOfElements,
                                                                           const Idx elementIdxShift,
                                                                           const unsigned int dimIndex = 0u) {
    // Check dimension
    //static_assert(alpaka::Dim<TAcc>::value == Dim1D::value,
    //              "Accelerator and maxNumberOfElements need to have same dimension.");
    auto [firstElementIdxLocal, endElementIdxLocal] = element_index_range_in_block(acc, elementIdxShift, dimIndex);

    // Truncate
    endElementIdxLocal = std::min(endElementIdxLocal, maxNumberOfElements);

    // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
    return {firstElementIdxLocal, endElementIdxLocal};
  }

  /*
   * Computes the range of the elements indexes in grid.
   * Warning: the max index is not truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid(const TAcc& acc,
                                                                Idx elementIdxShift,
                                                                const unsigned int dimIndex = 0u) {
    // Take into account the block index in grid.
    const Idx blockIndex(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
    const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

    // Shift to get global indices in grid (instead of local to the block)
    elementIdxShift += blockIndex * blockDimension;

    // Return element indexes, shifted by elementIdxShift.
    return element_index_range_in_block(acc, elementIdxShift, dimIndex);
  }

  /*
   * Computes the range of the elements indexes in grid.
   * Truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid_truncated(const TAcc& acc,
                                                                          const Idx maxNumberOfElements,
                                                                          Idx elementIdxShift,
                                                                          const unsigned int dimIndex = 0u) {
    // Check dimension
    //static_assert(dimIndex <= alpaka::Dim<TAcc>::value,
    //"Accelerator and maxNumberOfElements need to have same dimension.");
    auto [firstElementIdxGlobal, endElementIdxGlobal] = element_index_range_in_grid(acc, elementIdxShift, dimIndex);

    // Truncate
    endElementIdxGlobal = std::min(endElementIdxGlobal, maxNumberOfElements);

    // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
    return {firstElementIdxGlobal, endElementIdxGlobal};
  }

  /*
   * Computes the range of the element(s) index(es) in grid.
   * Truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid_truncated(const TAcc& acc,
                                                                          const Idx maxNumberOfElements,
                                                                          const unsigned int dimIndex = 0u) {
    Idx elementIdxShift = 0u;
    return element_index_range_in_grid_truncated(acc, maxNumberOfElements, elementIdxShift, dimIndex);
  }

  /*********************************************
   *           LOOP ON ALL ELEMENTS
   ********************************************/

  /*
   * Loop on all (CPU) elements.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Indexes are local to the BLOCK.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block(const TAcc& acc,
                                               const Idx maxNumberOfElements,
                                               const Idx elementIdxShift,
                                               const Func func,
                                               const unsigned int dimIndex = 0) {
    const auto& [firstElementIdx, endElementIdx] =
        element_index_range_in_block_truncated(acc, maxNumberOfElements, elementIdxShift, dimIndex);

    for (Idx elementIdx = firstElementIdx; elementIdx < endElementIdx; ++elementIdx) {
      func(elementIdx);
    }
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block(const TAcc& acc,
                                               const Idx maxNumberOfElements,
                                               const Func func,
                                               const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_block(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /*
   * Loop on all (CPU) elements.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Indexes are expressed in GRID 'frame-of-reference'.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid(const TAcc& acc,
                                              const Idx maxNumberOfElements,
                                              Idx elementIdxShift,
                                              const Func func,
                                              const unsigned int dimIndex = 0) {
    // Take into account the block index in grid to compute the element indices.
    const Idx blockIndex(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
    const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);
    elementIdxShift += blockIndex * blockDimension;

    for_each_element_in_block(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid(const TAcc& acc,
                                              const Idx maxNumberOfElements,
                                              const Func func,
                                              const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_grid(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /**************************************************************
   *          LOOP ON ALL ELEMENTS, WITH STRIDED ACCESS
   **************************************************************/

  /*
   * (CPU) Loop on all elements + (CPU/GPU) Strided access.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Stride to full problem size, by BLOCK size.
   * Indexes are local to the BLOCK.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block_strided(const TAcc& acc,
                                                       const Idx maxNumberOfElements,
                                                       const Idx elementIdxShift,
                                                       const Func func,
                                                       const unsigned int dimIndex = 0) {
    // Get thread / element indices in block.
    const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
        element_index_range_in_block(acc, elementIdxShift, dimIndex);

    // Stride = block size.
    const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

    // Strided access.
    for (Idx threadIdx = firstElementIdxNoStride, endElementIdx = endElementIdxNoStride;
         threadIdx < maxNumberOfElements;
         threadIdx += blockDimension, endElementIdx += blockDimension) {
      // (CPU) Loop on all elements.
      if (endElementIdx > maxNumberOfElements) {
        endElementIdx = maxNumberOfElements;
      }
      for (Idx i = threadIdx; i < endElementIdx; ++i) {
        func(i);
      }
    }
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block_strided(const TAcc& acc,
                                                       const Idx maxNumberOfElements,
                                                       const Func func,
                                                       const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_block_strided(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /*
   * (CPU) Loop on all elements + (CPU/GPU) Strided access.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Stride to full problem size, by GRID size.
   * Indexes are local to the GRID.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid_strided(const TAcc& acc,
                                                      const Idx maxNumberOfElements,
                                                      const Idx elementIdxShift,
                                                      const Func func,
                                                      const unsigned int dimIndex = 0) {
    // Get thread / element indices in block.
    const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
        element_index_range_in_grid(acc, elementIdxShift, dimIndex);

    // Stride = grid size.
    const Idx gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[dimIndex]);

    // Strided access.
    for (Idx threadIdx = firstElementIdxNoStride, endElementIdx = endElementIdxNoStride;
         threadIdx < maxNumberOfElements;
         threadIdx += gridDimension, endElementIdx += gridDimension) {
      // (CPU) Loop on all elements.
      if (endElementIdx > maxNumberOfElements) {
        endElementIdx = maxNumberOfElements;
      }
      for (Idx i = threadIdx; i < endElementIdx; ++i) {
        func(i);
      }
    }
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid_strided(const TAcc& acc,
                                                      const Idx maxNumberOfElements,
                                                      const Func func,
                                                      const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_grid_strided(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /**************************************************************
   *          LOOP ON ALL ELEMENTS WITH ONE LOOP
   **************************************************************/

  /*
   * Case where the input index i has reached the end of threadDimension: strides the input index.
   * Otherwise: do nothing.
   * NB 1: This helper function is used as a trick to only have one loop (like in legacy), instead of 2 loops
   * (like in all the other Alpaka helpers, 'for_each_element_in_block_strided' for example,
   * because of the additional loop over elements in Alpaka model).
   * This allows to keep the 'continue' and 'break' statements as-is from legacy code,
   * and hence avoids a lot of legacy code reshuffling.
   * NB 2: Modifies i, firstElementIdx and endElementIdx.
   */
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool next_valid_element_index_strided(
      Idx& i, Idx& firstElementIdx, Idx& endElementIdx, const Idx stride, const Idx maxNumberOfElements) {
    bool isNextStrideElementValid = true;
    if (i == endElementIdx) {
      firstElementIdx += stride;
      endElementIdx += stride;
      i = firstElementIdx;
      if (i >= maxNumberOfElements) {
        isNextStrideElementValid = false;
      }
    }
    return isNextStrideElementValid;
  }

  /*
   * Class which simplifies "for" loops over elements index
   */
  template <typename T_Acc, typename T>
  class elements_with_stride {
  public:
    ALPAKA_FN_ACC elements_with_stride(const T_Acc& acc,
                                       T extent,
                                       Idx elementIdxShift = 0,
                                       const unsigned int dimIndex = 0) {
      const Idx threadIndex = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndex];
      const Idx blockIndex = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex];
      const Idx blockDimension = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex];
      const Idx gridDimension = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex];

      thread_ = blockDimension * blockIndex + threadIndex;
      thread_ = thread_ + elementIdxShift;  // Add the shift
      stride_ = gridDimension * blockDimension;
      blockDim_ = blockDimension;

      extent_ = extent;
    }

    ALPAKA_FN_ACC elements_with_stride(const T_Acc& acc) {
      const Idx gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0]);
      elements_with_stride(acc, gridDimension);
    }

    class iterator {
      friend class elements_with_stride;

    public:
      ALPAKA_FN_ACC constexpr T operator*() const { return index_; }

      ALPAKA_FN_ACC constexpr iterator& operator++() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // CUDA backend: iterate over all the elements with a grid-wise stride
        index_ += stride_;
        if (index_ < extent_) {
          return *this;
        }
#else
        // CPU backend: iterate over all the elements for one thread
        index_ += 1;
        if (index_ < old_index_ + blockDim_ and index_ < extent_) {
          return *this;
        }
#endif
        // the iterator has reached or ovrflowed the end of the extent, clamp it to the extent
        index_ = extent_;
        return *this;
      }

      ALPAKA_FN_ACC constexpr iterator operator++(int) {
        iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC constexpr bool operator==(iterator const& other) const { return (index_ == other.index_); }

      ALPAKA_FN_ACC constexpr bool operator!=(iterator const& other) const { return index_ < other.index_; }

    private:
      ALPAKA_FN_ACC constexpr iterator(T thread, T stride, T extent, T blockDim)
          : thread_{thread}, stride_{stride}, extent_{extent}, index_{thread_}, old_index_{index_}, blockDim_{blockDim} {}

      ALPAKA_FN_ACC constexpr iterator(T thread, T stride, T extent, T index, T blockDim)
          : thread_{thread}, stride_{stride}, extent_{extent}, index_{index}, old_index_{index_}, blockDim_{blockDim} {}

      T thread_;
      T stride_;
      T extent_;
      T index_;
      T old_index_;
      T blockDim_;
    };

    ALPAKA_FN_ACC constexpr iterator begin() const { return iterator(thread_, stride_, extent_, blockDim_); }

    ALPAKA_FN_ACC constexpr iterator end() const { return iterator(thread_, stride_, extent_, extent_, blockDim_); }

  private:
    T thread_;
    T stride_;
    T extent_;
    T blockDim_;
  };

  /*
   * Class which simplifies "for" loops over elements index
   * Iterates over one dimension
   */
  template <typename T_Acc>
  class elements_with_stride_1d {
  public:
    ALPAKA_FN_ACC elements_with_stride_1d(const T_Acc& acc) {
      const Vec3D threadIndex(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
      const Vec3D blockIndex(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));

      const Vec3D blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
      const Vec3D gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));

      thread_ = {blockDimension[0u] * blockIndex[0u] + threadIndex[0u],
                 blockDimension[1u] * blockIndex[1u] + threadIndex[1u],
                 blockDimension[2u] * blockIndex[2u] + threadIndex[2u]};
      stride_ = {blockDimension[0u] * gridDimension[0u], 1, 1};
      extent_ = stride_;

      blockDim_ = blockDimension;
    }

    ALPAKA_FN_ACC elements_with_stride_1d(const T_Acc& acc, Vec3D extent, Vec3D elementIdxShift = Vec3D::all(0))
        : extent_(extent + elementIdxShift) {
      const Vec3D threadIndex(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
      const Vec3D blockIndex(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
      const Vec3D blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
      const Vec3D gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));

      thread_ = {blockDimension[0u] * blockIndex[0u] + threadIndex[0u],
                 blockDimension[1u] * blockIndex[1u] + threadIndex[1u],
                 blockDimension[2u] * blockIndex[2u] + threadIndex[2u]};
      thread_ = thread_ + elementIdxShift;
      stride_ = {blockDimension[0u] * gridDimension[0u], 1, 1};

      blockDim_ = blockDimension;
    }

    class iterator {
      friend class elements_with_stride_1d;

    public:
      ALPAKA_FN_ACC Vec3D operator*() const { return index_; }

      ALPAKA_FN_ACC constexpr iterator& operator++() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // increment the first coordinate
        index_[0u] += stride_[0u];
        if (index_[0u] < extent_[0u])
          return *this;
#else
        // increment the 3rd index and check its value
        index_[2u] += 1;
        if (index_[2u] == old_index_[2u] + blockDim_[2u])
          index_[2u] = old_index_[2u];

        //  if the 3rd index was reset, increment the 2nd index
        if (index_[2u] == old_index_[2u])
          index_[1u] += 1;
        if (index_[1u] == old_index_[1u] + blockDim_[1u])
          index_[1u] = old_index_[1u];

        // if the 3rd and 2nd indices were set, increment the first coordinate
        if (index_[1u] == old_index_[1u] and index_[2u] == old_index_[2u])
          index_[0u] += 1;

        if (index_[0u] < old_index_[0u] + blockDim_[0u] and index_[0u] < extent_[0u]) {
          return *this;
        }
#endif

        // the iterator has reached or ovrflowed the end of the extent, clamp it
        // to the extent
        index_ = extent_;
        return *this;
      }

      ALPAKA_FN_ACC constexpr iterator operator++(int) {
        iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC constexpr bool operator==(iterator const& other) const { return (index_ == other.index_); }

      ALPAKA_FN_ACC constexpr bool operator!=(iterator const& other) const { return index_[0u] < other.index_[0u]; }

    private:
      ALPAKA_FN_ACC iterator(Vec3D thread, Vec3D stride, Vec3D extent, Vec3D blockDim)
          : thread_{thread}, stride_{stride}, extent_{extent}, index_{thread_}, old_index_{index_}, blockDim_{blockDim} {}

      ALPAKA_FN_ACC iterator(Vec3D thread, Vec3D stride, Vec3D extent, Vec3D index, Vec3D blockDim)
          : thread_{thread}, stride_{stride}, extent_{extent}, index_{index}, old_index_{index_}, blockDim_{blockDim} {}

      Vec3D thread_;
      Vec3D stride_;
      Vec3D extent_;
      Vec3D index_;
      Vec3D old_index_;
      Vec3D blockDim_;
    };

    ALPAKA_FN_ACC constexpr iterator begin() const { return iterator(thread_, stride_, extent_, blockDim_); }

    ALPAKA_FN_ACC constexpr iterator end() const { return iterator(thread_, stride_, extent_, extent_, blockDim_); }

  private:
    Vec3D thread_ = Vec3D::all(1);
    Vec3D stride_ = Vec3D::all(1);
    Vec3D extent_ = Vec3D::all(1);
    Vec3D blockDim_ = Vec3D::all(1);
  };

  /*
   * Class which simplifies "for" loops over elements index
   * Iterates over two dimensions
   */
  template <typename T_Acc>
  class elements_with_stride_2d {
  public:
    ALPAKA_FN_ACC elements_with_stride_2d(const T_Acc& acc) {
      const Vec3D threadIndex(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
      const Vec3D blockIndex(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
      const Vec3D blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
      const Vec3D gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));

      thread_ = {blockDimension[0u] * blockIndex[0u] + threadIndex[0u],
                 blockDimension[1u] * blockIndex[1u] + threadIndex[1u],
                 blockDimension[2u] * blockIndex[2u] + threadIndex[2u]};
      stride_ = {blockDimension[0u] * gridDimension[0u], blockDimension[1u] * gridDimension[1u], 1};
      extent_ = stride_;

      blockDim_ = blockDimension;
    }

    ALPAKA_FN_ACC elements_with_stride_2d(const T_Acc& acc, Vec3D extent, Vec3D elementIdxShift = Vec3D::all(0))
        : extent_(extent + elementIdxShift) {
      const Vec3D threadIndex(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
      const Vec3D blockIndex(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
      const Vec3D blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
      const Vec3D gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));

      thread_ = {blockDimension[0u] * blockIndex[0u] + threadIndex[0u],
                 blockDimension[1u] * blockIndex[1u] + threadIndex[1u],
                 blockDimension[2u] * blockIndex[2u] + threadIndex[2u]};
      thread_ = thread_ + elementIdxShift;
      stride_ = {blockDimension[0u] * gridDimension[0u], blockDimension[1u] * gridDimension[1u], 1};

      blockDim_ = blockDimension;
    }

    class iterator {
      friend class elements_with_stride_2d;

    public:
      ALPAKA_FN_ACC Vec3D operator*() const { return index_; }

      ALPAKA_FN_ACC constexpr iterator& operator++() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // increment the first coordinate
        index_[0u] += stride_[0u];
        if (index_[0u] < extent_[0u])
          return *this;

        // if the first coordinate overflows, reset it and increment the second
        // coordinate
        index_[0u] = thread_[0u];
        index_[1u] += stride_[1u];
        if (index_[1u] < extent_[1u])
          return *this;
#else
        // increment the 3rd index and check its value
        index_[2u] += 1;
        if (index_[2u] == old_index_[2u] + blockDim_[2u])
          index_[2u] = old_index_[2u];

        //  if the 3rd index was reset, increment the 2nd index
        if (index_[2u] == old_index_[2u])
          index_[1u] += 1;
        if (index_[1u] == old_index_[1u] + blockDim_[1u] || index_[1u] == extent_[1u])
          index_[1u] = old_index_[1u];

        // if the 3rd and 2nd indices were set, increment the first coordinate
        if (index_[1u] == old_index_[1u] and index_[2u] == old_index_[2u])
          index_[0u] += 1;

        if (index_[0u] < old_index_[0u] + blockDim_[0u] and index_[0u] < extent_[0u] and index_[1u] < extent_[1u]) {
          return *this;
        }
#endif

        // the iterator has reached or ovrflowed the end of the extent, clamp it
        // to the extent
        index_ = extent_;
        return *this;
      }

      ALPAKA_FN_ACC constexpr iterator operator++(int) {
        iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC constexpr bool operator==(iterator const& other) const { return (index_ == other.index_); }

      ALPAKA_FN_ACC constexpr bool operator!=(iterator const& other) const {
        return (index_[0u] < other.index_[0u] and index_[1u] < other.index_[1u]);
      }

    private:
      ALPAKA_FN_ACC iterator(Vec3D thread, Vec3D stride, Vec3D extent, Vec3D blockDim)
          : thread_{thread}, stride_{stride}, extent_{extent}, index_{thread_}, old_index_{index_}, blockDim_{blockDim} {}

      ALPAKA_FN_ACC iterator(Vec3D thread, Vec3D stride, Vec3D extent, Vec3D index, Vec3D blockDim)
          : thread_{thread}, stride_{stride}, extent_{extent}, index_{index}, old_index_{index_}, blockDim_{blockDim} {}

      Vec3D thread_;
      Vec3D stride_;
      Vec3D extent_;
      Vec3D index_;
      Vec3D old_index_;
      Vec3D blockDim_;
    };

    ALPAKA_FN_ACC constexpr iterator begin() const { return iterator(thread_, stride_, extent_, blockDim_); }

    ALPAKA_FN_ACC constexpr iterator end() const { return iterator(thread_, stride_, extent_, extent_, blockDim_); }

  private:
    Vec3D thread_ = Vec3D::all(1);
    Vec3D stride_ = Vec3D::all(1);
    Vec3D extent_ = Vec3D::all(1);
    Vec3D blockDim_ = Vec3D::all(1);
  };

  /*
   * Class which simplifies "for" loops over elements index
   * Iterates over three dimensions
   */
  template <typename T_Acc>
  class elements_with_stride_3d {
  public:
    ALPAKA_FN_ACC elements_with_stride_3d(const T_Acc& acc) {
      const Vec3D threadIndex(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
      const Vec3D blockIndex(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
      const Vec3D blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
      const Vec3D gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));

      thread_ = {blockDimension[0u] * blockIndex[0u] + threadIndex[0u],
                 blockDimension[1u] * blockIndex[1u] + threadIndex[1u],
                 blockDimension[2u] * blockIndex[2u] + threadIndex[2u]};
      stride_ = {blockDimension[0u] * gridDimension[0u],
                 blockDimension[1u] * gridDimension[1u],
                 blockDimension[2u] * gridDimension[2u]};
      extent_ = stride_;

      blockDim_ = blockDimension;
    }

    ALPAKA_FN_ACC elements_with_stride_3d(const T_Acc& acc, Vec3D extent, Vec3D elementIdxShift = Vec3D::all(0))
        : extent_(extent + elementIdxShift) {
      const Vec3D threadIndex(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc));
      const Vec3D blockIndex(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
      const Vec3D blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
      const Vec3D gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));

      thread_ = {blockDimension[0u] * blockIndex[0u] + threadIndex[0u],
                 blockDimension[1u] * blockIndex[1u] + threadIndex[1u],
                 blockDimension[2u] * blockIndex[2u] + threadIndex[2u]};
      thread_ = thread_ + elementIdxShift;
      stride_ = {blockDimension[0u] * gridDimension[0u],
                 blockDimension[1u] * gridDimension[1u],
                 blockDimension[2u] * gridDimension[2u]};

      blockDim_ = blockDimension;
    }

    class iterator {
      friend class elements_with_stride_3d;

    public:
      ALPAKA_FN_ACC Vec3D operator*() const { return index_; }

      ALPAKA_FN_ACC constexpr iterator& operator++() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // increment the first coordinate
        index_[0u] += stride_[0u];
        if (index_[0u] < extent_[0u])
          return *this;

        // if the first coordinate overflows, reset it and increment the second
        // coordinate
        index_[0u] = thread_[0u];
        index_[1u] += stride_[1u];
        if (index_[1u] < extent_[1u])
          return *this;

        // if the second coordinate overflows, reset it and increment the third
        // coordinate
        index_[1u] = thread_[1u];
        index_[2u] += stride_[2u];
        if (index_[2u] < extent_[2u])
          return *this;
#else
        // increment the 3rd index and check its value
        index_[2u] += 1;
        if (index_[2u] == old_index_[2u] + blockDim_[2u] || index_[2u] == extent_[2u])
          index_[2u] = old_index_[2u];

        //  if the 3rd index was reset, increment the 2nd index
        if (index_[2u] == old_index_[2u])
          index_[1u] += 1;
        if (index_[1u] == old_index_[1u] + blockDim_[1u] || index_[1u] == extent_[1u])
          index_[1u] = old_index_[1u];

        // if the 3rd and 2nd indices were set, increment the first coordinate
        if (index_[1u] == old_index_[1u] and index_[2u] == old_index_[2u])
          index_[0u] += 1;
        if (index_[0u] < old_index_[0u] + blockDim_[0u] and index_[0u] < extent_[0u] and index_[1u] < extent_[1u] and
            index_[2u] < extent_[2u]) {
          return *this;
        }
#endif

        // the iterator has reached or ovrflowed the end of the extent, clamp it
        // to the extent
        index_ = extent_;
        return *this;
      }

      ALPAKA_FN_ACC constexpr iterator operator++(int) {
        iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC constexpr bool operator==(iterator const& other) const { return (index_ == other.index_); }

      ALPAKA_FN_ACC constexpr bool operator!=(iterator const& other) const {
        return (index_[0u] < other.index_[0u] and index_[1u] < other.index_[1u] and index_[2u] < other.index_[2u]);
      }

    private:
      ALPAKA_FN_ACC iterator(Vec3D thread, Vec3D stride, Vec3D extent, Vec3D blockDim)
          : thread_{thread}, stride_{stride}, extent_{extent}, index_{thread_}, old_index_{index_}, blockDim_{blockDim} {}

      ALPAKA_FN_ACC iterator(Vec3D thread, Vec3D stride, Vec3D extent, Vec3D index, Vec3D blockDim)
          : thread_{thread}, stride_{stride}, extent_{extent}, index_{index}, old_index_{index_}, blockDim_{blockDim} {}

      Vec3D thread_;
      Vec3D stride_;
      Vec3D extent_;
      Vec3D index_;
      Vec3D old_index_;
      Vec3D blockDim_;
    };

    ALPAKA_FN_ACC constexpr iterator begin() const { return iterator(thread_, stride_, extent_, blockDim_); }

    ALPAKA_FN_ACC constexpr iterator end() const { return iterator(thread_, stride_, extent_, extent_, blockDim_); }

  private:
    Vec3D thread_ = Vec3D::all(1);
    Vec3D stride_ = Vec3D::all(1);
    Vec3D extent_ = Vec3D::all(1);
    Vec3D blockDim_ = Vec3D::all(1);
  };

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // AlpakaCore_alpakaWorkDivHelper_h
