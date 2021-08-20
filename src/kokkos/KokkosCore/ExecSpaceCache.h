#ifndef KokkosCore_ExecSpaceCache_h
#define KokkosCore_ExecSpaceCache_h

#include <memory>

#include <Kokkos_Core.hpp>

#include "Framework/ReusableObjectHolder.h"

namespace cms {
  namespace kokkos {
    template <typename ExecSpace>
    class ExecSpaceWrapper {
    public:
      ExecSpaceWrapper() = default;

      ExecSpace const& space() const { return space_; }

    private:
      ExecSpace space_;
    };

    template <typename ExecSpace>
    class ExecSpaceCache {
    public:
      using CacheType = edm::ReusableObjectHolder<ExecSpaceWrapper<ExecSpace>>;

      ExecSpaceCache() : cache_{std::make_unique<CacheType>()} {}

      // Gets a (cached) execution space object. The object will be
      // returned to the cache by the shared_ptr destructor.  This
      // function is thread safe
      std::shared_ptr<ExecSpaceWrapper<ExecSpace>> get() {
        return cache_->makeOrGet([]() { return std::make_unique<ExecSpaceWrapper<ExecSpace>>(); });
      }

      // Need to be able to clear before the destruction of globals
      // because the execution space objects need to be destructed
      // before the destruction of Kokkos runtime. In addition, these
      // objects need to be destructed before the
      // cms::cuda::StreamCache is destroyed.
      void clear() { cache_.reset(); }

    private:
      std::unique_ptr<CacheType> cache_;
    };

    // Gets the global instance of a ExecSpaceCache
    // This function is thread safe
    template <typename ExecSpace>
    ExecSpaceCache<ExecSpace>& getExecSpaceCache() {
      // the public interface is thread safe
      static ExecSpaceCache<ExecSpace> cache;
      return cache;
    }
  }  // namespace kokkos
}  // namespace cms

#ifdef KOKKOS_ENABLE_CUDA
#include "CUDACore/StreamCache.h"
namespace cms {
  namespace kokkos {
    template <>
    class ExecSpaceWrapper<Kokkos::Cuda> {
    public:
      ExecSpaceWrapper(Kokkos::Cuda space, cms::cuda::SharedStreamPtr stream)
          : space_(std::move(space)), stream_(std::move(stream)) {}

      Kokkos::Cuda const& space() const { return space_; }
      cudaStream_t stream() const { return stream_.get(); }

    private:
      Kokkos::Cuda space_;
      cms::cuda::SharedStreamPtr stream_;
    };

    template <>
    inline std::shared_ptr<ExecSpaceWrapper<Kokkos::Cuda>> ExecSpaceCache<Kokkos::Cuda>::get() {
      return cache_->makeOrGet([]() {
        auto streamPtr = cms::cuda::getStreamCache().get();
        return std::make_unique<ExecSpaceWrapper<Kokkos::Cuda>>(Kokkos::Cuda(streamPtr.get()), std::move(streamPtr));
      });
    }
  }  // namespace kokkos
}  // namespace cms

#endif

#endif
