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
      ExecSpaceWrapper(ExecSpace space) : space_(std::move(space)) {}

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
        return cache_->makeOrGet([]() {
                auto instances = Kokkos::Experimental::partition_space(ExecSpace(), 1);
                return std::make_unique<ExecSpaceWrapper<ExecSpace>>(instances[0]); });
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



#endif
