#include "alpakaAlgo1.h"

namespace {
  constexpr unsigned int NUM_VALUES = 1000;


  struct vectorAdd {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(
				  T_Acc const& acc, const T_Data* __restrict__ a, const T_Data* __restrict__ b, T_Data* __restrict__ c, unsigned int numElements) const {

      // Global thread index in grid
      const uint32_t threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
      const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

      // Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
      const uint32_t firstElementIdxGlobal = threadIdxGlobal * threadDimension;
      const uint32_t endElementIdxGlobalUncut = firstElementIdxGlobal + threadDimension;
      const uint32_t endElementIdxGlobal = std::min(endElementIdxGlobalUncut, numElements);


      for (uint32_t i = firstElementIdxGlobal; i < endElementIdxGlobal; ++i) {
	c[i] = a[i] + b[i];
      }

    }
  };


  struct vectorProd {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(
				  T_Acc const& acc, const T_Data* __restrict__ a, const T_Data* __restrict__ b, T_Data* __restrict__ c, unsigned int numElements) const {

      // Global thread index in Dim2 grid
      const auto& threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
      const uint32_t threadIdxGlobalX(threadIdxGlobal[0u]);
      const uint32_t threadIdxGlobalY(threadIdxGlobal[1u]);

      // Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
      const auto& threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
      const uint32_t threadDimensionX(threadDimension[0u]);
      const uint32_t firstElementIdxGlobalX = threadIdxGlobalX * threadDimensionX;
      const uint32_t endElementIdxGlobalXUncut = firstElementIdxGlobalX + threadDimensionX;
      const uint32_t endElementIdxGlobalX = std::min(endElementIdxGlobalXUncut, numElements);

      const uint32_t threadDimensionY(threadDimension[1u]);
      const uint32_t firstElementIdxGlobalY = threadIdxGlobalY * threadDimensionY;
      const uint32_t endElementIdxGlobalYUncut = firstElementIdxGlobalY + threadDimensionY;
      const uint32_t endElementIdxGlobalY = std::min(endElementIdxGlobalYUncut, numElements);


      for (uint32_t col = firstElementIdxGlobalX; col < endElementIdxGlobalX; ++col) {
	for (uint32_t row = firstElementIdxGlobalY; row < endElementIdxGlobalY; ++row) {

	  c[row + numElements * col] = a[row] * b[col];
	}
      }

    }
  };


  struct matrixMul {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(
				  T_Acc const& acc, const T_Data* __restrict__ a, const T_Data* __restrict__ b, T_Data* __restrict__ c, unsigned int numElements) const {

      // Global thread index in Dim2 grid
      const auto& threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
      const uint32_t threadIdxGlobalX(threadIdxGlobal[0u]);
      const uint32_t threadIdxGlobalY(threadIdxGlobal[1u]);

      // Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
      const auto& threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
      const uint32_t threadDimensionX(threadDimension[0u]);
      const uint32_t firstElementIdxGlobalX = threadIdxGlobalX * threadDimensionX;
      const uint32_t endElementIdxGlobalXUncut = firstElementIdxGlobalX + threadDimensionX;
      const uint32_t endElementIdxGlobalX = std::min(endElementIdxGlobalXUncut, numElements);

      const uint32_t threadDimensionY(threadDimension[1u]);
      const uint32_t firstElementIdxGlobalY = threadIdxGlobalY * threadDimensionY;
      const uint32_t endElementIdxGlobalYUncut = firstElementIdxGlobalY + threadDimensionY;
      const uint32_t endElementIdxGlobalY = std::min(endElementIdxGlobalYUncut, numElements);


      for (uint32_t col = firstElementIdxGlobalX; col < endElementIdxGlobalX; ++col) {
	for (uint32_t row = firstElementIdxGlobalY; row < endElementIdxGlobalY; ++row) {	

	  T_Data tmp = 0;
	  for (unsigned int i = 0; i < numElements; ++i) {
	    tmp += a[row + numElements * i] * b[i + numElements * col];
	  }
	  c[row + numElements * col] = tmp;
	}
      }

    }
  };


  struct matrixMulVector {
    template <typename T_Acc, typename T_Data>
    ALPAKA_FN_ACC void operator()(
				  T_Acc const& acc, const T_Data* __restrict__ a, const T_Data* __restrict__ b, T_Data* __restrict__ c, unsigned int numElements) const {

      // Global thread index in grid
      const uint32_t threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
      const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

      // Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
      const uint32_t firstElementIdxGlobal = threadIdxGlobal * threadDimension;
      const uint32_t endElementIdxGlobalUncut = firstElementIdxGlobal + threadDimension;
      const uint32_t endElementIdxGlobal = std::min(endElementIdxGlobalUncut, numElements);


      for (uint32_t row = firstElementIdxGlobal; row < endElementIdxGlobal; ++row) {
	T_Data tmp = 0;
	for (unsigned int i = 0; i < numElements; ++i) {
	  tmp += a[row * numElements + i] * b[i];
	}
	c[row] = tmp;
      }

    }
  };



  /* 
     DEBUG ONLY
     Obviously not optimized (and contains printf anyway), incorporated to verify results.
  */
  namespace debug {
    constexpr float TOLERANCE_RATIO = 0.01;


    struct verifyVectorAdd {
      template <typename T_Acc, typename T_Data>
      ALPAKA_FN_ACC void operator()(
				    const T_Acc& acc, const T_Data* result, unsigned int numElements) const {

	// Global thread index in grid
	const uint32_t threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
	const uint32_t threadDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

	// Global element index (obviously relevant for CPU only, for GPU, i = threadIndexGlobal only)
	const uint32_t firstElementIdxGlobal = threadIdxGlobal * threadDimension;
	const uint32_t endElementIdxGlobalUncut = firstElementIdxGlobal + threadDimension;
	const uint32_t endElementIdxGlobal = std::min(endElementIdxGlobalUncut, numElements);


	for (uint32_t i = firstElementIdxGlobal; i < endElementIdxGlobal; ++i) {

	  // theoreticalResult = i+i^2 = i*(i+1)
	  if (result[i] != i*(i+1)) { 
	    printf("Wrong vectorAdd results, i = %u, c[i] = %f.\n",
		   i, result[i]
		   );
	  }
	}
	
      }
    };


    struct verifyVectorProd {
      template <typename T_Acc, typename T_Data>
      ALPAKA_FN_ACC void operator()(
				    const T_Acc& acc, const T_Data* result, unsigned int numElements) const {

	const auto& threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
	const uint32_t threadIdxGlobalX(threadIdxGlobal[0u]);
	const uint32_t threadIdxGlobalY(threadIdxGlobal[1u]);


	if (threadIdxGlobalX == 0 && threadIdxGlobalY == 0) {
	  for (unsigned int row = 0; row < numElements; ++row) {
	    for (unsigned int col = 0; col < numElements; ++col) {

	      const T_Data theoreticalResult = static_cast<T_Data>(row)*col*col;
	      const T_Data diff = result[row + numElements * col] - theoreticalResult;
	      const T_Data tolerance = theoreticalResult * TOLERANCE_RATIO;
	      if (diff > tolerance || diff < -tolerance) {
		printf("Wrong vectorProd results, row = %u, col = %u, c[row,col] = %f.\n",
		       row, col, result[row * numElements + col]
		       );
	      }
	    }
	  }
	}

      }
    };


    struct verifyMatrixMul {
      template <typename T_Acc, typename T_Data>
      ALPAKA_FN_ACC void operator()(
				    const T_Acc& acc, const T_Data* result, unsigned int numElements) const {

	const auto& threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
	const uint32_t threadIdxGlobalX(threadIdxGlobal[0u]);
	const uint32_t threadIdxGlobalY(threadIdxGlobal[1u]);


	if (threadIdxGlobalX == 0 && threadIdxGlobalY == 0) {
	  const T_Data partialtheoreticalResult = static_cast<T_Data>(numElements-1)*(numElements-1)*numElements*numElements/4;
	
	  for (unsigned int row = 0; row < numElements; ++row) {
	    for (unsigned int col = 0; col < numElements; ++col) {

	      // theoreticalResult = row * col * (col+1) * Sum(k=1 to numElements-1, k^3)
	      const T_Data theoreticalResult = static_cast<T_Data>(row)*col*(col+1)*partialtheoreticalResult;
	      const T_Data diff = result[row + numElements * col] - theoreticalResult;
	      const T_Data tolerance = theoreticalResult * TOLERANCE_RATIO;
	      
	      if (diff > tolerance || diff < -tolerance) {
		printf("Wrong matrix multiplication results, row = %u, col = %u, c[row,col] = %f, theoreticalResult = %f.\n",
		       row, col, result[row * numElements + col], theoreticalResult
		       );
	      }
	    }
	  }
	}

      }
    };


    struct verifyMatrixMulVector {
      template <typename T_Acc, typename T_Data>
      ALPAKA_FN_ACC void operator()(
				    const T_Acc& acc, const T_Data* result, unsigned int numElements) const {

	const uint32_t threadIdxGlobal(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

	if (threadIdxGlobal == 0) {
	  const unsigned long int N = numElements - 1;
	  const unsigned long int N2 = N*N;
	  const unsigned long int N3 = N2 * N;
	  const unsigned long int N4 = N2 * N2;
	  const unsigned long int N5 = N4 * N;
	  const T_Data partialtheoreticalResult = static_cast<T_Data>(N2)*(N+1)*(N+1)/4 * ((6*N5+15*N4+10*N3-N)/30 + (N+1)*(N+1)*N2/2 + N*(N+1)/2);


	  for (unsigned int i = 0; i < numElements; ++i) {

	    // theoreticalResult = N^2*(N+1)^2/4 * i * Sum(k=1 to N, k^2*(k+1)^2)
	    const T_Data theoreticalResult = i * partialtheoreticalResult;
	    const T_Data diff = result[i] - theoreticalResult;
	    const T_Data tolerance = theoreticalResult * TOLERANCE_RATIO;

	    if (diff > tolerance || diff < -tolerance) {
	      printf("Wrong matrix-vector multiplication results, i = %u, c[i] = %f, theoreticalResult = %f.\n",
		     i, result[i], theoreticalResult
		     );
	    
	    }
	  }
	}

      }
    };

  } // namespace debug


} // namespace



namespace ALPAKA_ACCELERATOR_NAMESPACE {
  void alpakaAlgo1() {

    const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    const DevAcc2 device(alpaka::pltf::getDevByIdx<PltfAcc2>(0u));
    const Vec size(NUM_VALUES);
    Queue queue(device);

   
    // Host data
    auto h_a_buf = alpaka::mem::buf::alloc<float, Idx>(host, size);
    auto h_b_buf = alpaka::mem::buf::alloc<float, Idx>(host, size);
    auto h_a = alpaka::mem::view::getPtrNative(h_a_buf);
    auto h_b = alpaka::mem::view::getPtrNative(h_b_buf);
    for (auto i = 0U; i < NUM_VALUES; i++) {
      h_a[i] = i;
      h_b[i] = i * i;
    }


    // Device data
    auto d_a_buf = alpaka::mem::buf::alloc<float, Idx>(device, size);
    auto d_b_buf = alpaka::mem::buf::alloc<float, Idx>(device, size);
    alpaka::mem::view::copy(queue, d_a_buf, h_a_buf, size);
    alpaka::mem::view::copy(queue, d_b_buf, h_b_buf, size);
    auto d_c_buf = alpaka::mem::buf::alloc<float, Idx>(device, size);


    // Prepare 1D workDiv
    Vec elementsPerThread(Vec::all(1));
    Vec threadsPerBlock(Vec::all(32));
    const Vec blocksPerGrid(Vec::all((NUM_VALUES + 32 - 1) / 32));
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED || \
  ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_BT_OMP4_ENABLED
    // on the GPU, run with 32 threads in parallel per block, each looking at a single element
    // on the CPU, run serially with a single thread per block, over 32 elements
    std::swap(threadsPerBlock, elementsPerThread);
#endif
    const WorkDiv workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);


    // VECTOR ADDITION
    alpaka::queue::enqueue(queue,
                           alpaka::kernel::createTaskKernel<Acc>(workDiv,
                                                                 vectorAdd(),
                                                                 alpaka::mem::view::getPtrNative(d_a_buf),
                                                                 alpaka::mem::view::getPtrNative(d_b_buf),
                                                                 alpaka::mem::view::getPtrNative(d_c_buf),
                                                                 NUM_VALUES));

    // Prepare 2D workDiv
    Vec2 elementsPerThread2(1u, 1u);
    const unsigned int threadsPerBlockSide = (NUM_VALUES < 32 ? NUM_VALUES : 32u);
    Vec2 threadsPerBlock2(threadsPerBlockSide, threadsPerBlockSide);
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED || \
  ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_BT_OMP4_ENABLED
    // on the GPU, run with 32 threads in parallel per block, each looking at a single element
    // on the CPU, run serially with a single thread per block, over 32 elements
    std::swap(threadsPerBlock2, elementsPerThread2);
#endif
    const unsigned int blocksPerGridSide = (NUM_VALUES <= 32 ? 1 : std::ceil(NUM_VALUES / 32.));
    const Vec2 blocksPerGrid2(blocksPerGridSide, blocksPerGridSide);
    const WorkDiv2 workDiv2(blocksPerGrid2, threadsPerBlock2, elementsPerThread2);


    // Device data
    const Vec2 sizeSquare(NUM_VALUES, NUM_VALUES);
    auto d_ma_buf = alpaka::mem::buf::alloc<float, Idx>(device, sizeSquare);
    auto d_mb_buf = alpaka::mem::buf::alloc<float, Idx>(device, sizeSquare);
    auto d_mc_buf = alpaka::mem::buf::alloc<float, Idx>(device, sizeSquare);


    // VECTOR MULTIPLICATION
    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<Acc2>(workDiv2,
								  vectorProd(),
								  alpaka::mem::view::getPtrNative(d_a_buf),
								  alpaka::mem::view::getPtrNative(d_b_buf),
								  alpaka::mem::view::getPtrNative(d_ma_buf),
								  NUM_VALUES));

    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<Acc2>(workDiv2,
								  vectorProd(),
								  alpaka::mem::view::getPtrNative(d_a_buf),
								  alpaka::mem::view::getPtrNative(d_c_buf),
								  alpaka::mem::view::getPtrNative(d_mb_buf),
								  NUM_VALUES));

    // MATRIX MULTIPLICATION
    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<Acc2>(workDiv2,
								  matrixMul(),
								  alpaka::mem::view::getPtrNative(d_ma_buf),
								  alpaka::mem::view::getPtrNative(d_mb_buf),
								  alpaka::mem::view::getPtrNative(d_mc_buf),
								  NUM_VALUES));

    // MATRIX - VECTOR MULTIPLICATION
    alpaka::queue::enqueue(queue,
			   alpaka::kernel::createTaskKernel<Acc>(workDiv,
								 matrixMulVector(),
								 alpaka::mem::view::getPtrNative(d_mc_buf),
								 alpaka::mem::view::getPtrNative(d_b_buf),
								 alpaka::mem::view::getPtrNative(d_c_buf),
								 NUM_VALUES));

    alpaka::wait::wait(queue);    
  }
} // namespace ALPAKA_ACCELERATOR_NAMESPACE
