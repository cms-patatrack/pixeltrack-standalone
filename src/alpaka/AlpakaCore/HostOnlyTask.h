#ifndef AlpakaCore_HostOnlyTask_h
#define AlpakaCore_HostOnlyTask_h

#include <functional>
#include <memory>

#include <alpaka/alpaka.hpp>
#include <alpaka/core/CallbackThread.hpp>

namespace alpaka {

  class HostOnlyTask {
  public:
    HostOnlyTask(std::function<void()> task) : task_(std::move(task)) {}

    void operator()() const { task_(); }

  private:
    std::function<void()> task_;
  };

  namespace trait {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    //! The CUDA async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueCudaRtNonBlocking, HostOnlyTask> {
      using TApi = ApiCudaRt;

      static void CUDART_CB callback(cudaStream_t /*queue*/, cudaError_t /*status*/, void* arg) {
        //ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(status);
        std::unique_ptr<HostOnlyTask> pTask(static_cast<HostOnlyTask*>(arg));
        (*pTask)();
      }

      ALPAKA_FN_HOST static auto enqueue(QueueCudaRtNonBlocking& queue, HostOnlyTask task) -> void {
        auto pTask = std::make_unique<HostOnlyTask>(std::move(task));
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            cudaStreamAddCallback(alpaka::getNativeHandle(queue), callback, static_cast<void*>(pTask.release()), 0u));
      }
    };
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    //! The HIP async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueHipRtNonBlocking, HostOnlyTask> {
      using TApi = ApiHipRt;

      static void callback(hipStream_t /*queue*/, hipError_t /*status*/, void* arg) {
        //ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(status);
        std::unique_ptr<HostOnlyTask> pTask(static_cast<HostOnlyTask*>(arg));
        (*pTask)();
      }

      ALPAKA_FN_HOST static auto enqueue(QueueHipRtNonBlocking& queue, HostOnlyTask task) -> void {
        auto pTask = std::make_unique<HostOnlyTask>(std::move(task));
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            hipStreamAddCallback(alpaka::getNativeHandle(queue), callback, static_cast<void*>(pTask.release()), 0u));
      }
    };
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_SYCL_ONEAPI_CPU
    using Task = std::packaged_task<void()>;
    //! The SYCL CPU async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueCpuSyclNonBlocking, Task> {
      ALPAKA_FN_HOST static auto enqueue(QueueCpuSyclNonBlocking& queue, Task&& task) -> void {
        alpaka::core::CallbackThread m_callbackThread;
        queue.getNativeHandle().wait();
        m_callbackThread.submit(std::forward<Task>(task));
      }
    };
#endif // ALPAKA_SYCL_ONEAPI_CPU

#ifdef ALPAKA_SYCL_ONEAPI_GPU
    using Task = std::packaged_task<void()>;
    //! The SYCL GPU async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueGpuSyclIntelNonBlocking, Task> {
      ALPAKA_FN_HOST static auto enqueue(QueueGpuSyclIntelNonBlocking& queue, Task&& task) -> void {
        alpaka::core::CallbackThread m_callbackThread;
        queue.getNativeHandle().wait();
        m_callbackThread.submit(std::forward<Task>(task));
      }
    };
#endif  // ALPAKA_SYCL_ONEAPI_GPU

  }  // namespace trait

}  // namespace alpaka

#endif  // AlpakaCore_HostOnlyTask_h
