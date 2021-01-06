#ifndef KokkosCore_ProductBase_h
#define KokkosCore_ProductBase_h

#include <exception>
#include <memory>
#include <string>

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include "CUDACore/EventCache.h"
#include "CUDACore/SharedEventPtr.h"
#include "CUDACore/SharedStreamPtr.h"
#include "CUDACore/StreamCache.h"
#endif
#include "Framework/WaitingTaskHolder.h"
#include "Framework/WaitingTaskWithArenaHolder.h"

namespace cms {
  namespace kokkos {
    namespace impl {
      class ExecSpaceSpecificBase {
      public:
        ExecSpaceSpecificBase() = default;
        virtual ~ExecSpaceSpecificBase();
      };

      template <typename ExecSpace>
      class ExecSpaceSpecific : public ExecSpaceSpecificBase {
      public:
        ExecSpaceSpecific() = default;
        ~ExecSpaceSpecific() override = default;

        std::unique_ptr<ExecSpaceSpecific> cloneShareStream() const {
          return std::make_unique<ExecSpaceSpecific>(*this);
        }

        std::unique_ptr<ExecSpaceSpecific> cloneShareAll() const { return std::make_unique<ExecSpaceSpecific>(*this); }

        void recordEvent() {}
        void enqueueCallback(edm::WaitingTaskWithArenaHolder withArenaHolder) {
          space_.fence();
          auto holder = withArenaHolder.makeWaitingTaskHolderAndRelease();
          holder.doneWaiting(nullptr);
        }
        void synchronizeWith(ExecSpaceSpecific const& other) const { other.execSpace().fence(); }

        ExecSpace const& execSpace() const { return space_; }

      private:
        ExecSpace space_;
      };

#ifdef KOKKOS_ENABLE_CUDA
      template <>
      class ExecSpaceSpecific<Kokkos::Cuda> : public ExecSpaceSpecificBase {
      public:
        ExecSpaceSpecific() : ExecSpaceSpecific(cms::cuda::getStreamCache().get()) {}
        explicit ExecSpaceSpecific(cms::cuda::SharedStreamPtr stream)
            : ExecSpaceSpecific(stream, cms::cuda::getEventCache().get()) {}
        explicit ExecSpaceSpecific(cms::cuda::SharedStreamPtr stream, cms::cuda::SharedEventPtr event)
            : space_(stream.get()), stream_(std::move(stream)), event_(std::move(event)) {}

        ~ExecSpaceSpecific() override = default;

        std::unique_ptr<ExecSpaceSpecific> cloneShareStream() const {
          return std::make_unique<ExecSpaceSpecific>(stream_);
        }

        std::unique_ptr<ExecSpaceSpecific> cloneShareAll() const {
          return std::make_unique<ExecSpaceSpecific>(stream_, event_);
        }

        void recordEvent() {
          // Intentionally not checking the return value to avoid throwing
          // exceptions. If this call would fail, we should get failures
          // elsewhere as well.
          cudaEventRecord(event_.get(), stream_.get());
        }

        void enqueueCallback(edm::WaitingTaskWithArenaHolder holder);

        void synchronizeWith(ExecSpaceSpecific const& other);

        int device() const { return space_.cuda_device(); }
        Kokkos::Cuda const& execSpace() const { return space_; }

      private:
        bool isAvailable() const;

        Kokkos::Cuda space_;
        cms::cuda::SharedStreamPtr stream_;
        cms::cuda::SharedEventPtr event_;
      };
#endif
    }  // namespace impl

    class ProductBase {
    public:
      ProductBase() = default;
      explicit ProductBase(std::unique_ptr<impl::ExecSpaceSpecificBase> spaceSpecific)
          : execSpaceSpecific_(std::move(spaceSpecific)) {}
      ~ProductBase() = default;

      ProductBase(const ProductBase&) = delete;
      ProductBase& operator=(const ProductBase&) = delete;
      ProductBase(ProductBase&&) = default;
      ProductBase& operator=(ProductBase&&) = default;

      bool isValid() const { return execSpaceSpecific_.get() != nullptr; }

      template <typename ExecSpace>
      impl::ExecSpaceSpecific<ExecSpace> const& execSpaceSpecific() const {
        auto const& sp = *execSpaceSpecific_;
        if (typeid(sp) != typeid(impl::ExecSpaceSpecific<ExecSpace>)) {
          throw std::runtime_error(std::string("Incompatible Execution space: has ") +
                                   typeid(sp).name() + ", but " +
                                   typeid(impl::ExecSpaceSpecific<ExecSpace>).name() + " was asked for");
        }
        return static_cast<impl::ExecSpaceSpecific<ExecSpace> const&>(sp);
      }

    private:
      std::unique_ptr<impl::ExecSpaceSpecificBase> execSpaceSpecific_;
    };
  }  // namespace kokkos
}  // namespace cms

#endif
