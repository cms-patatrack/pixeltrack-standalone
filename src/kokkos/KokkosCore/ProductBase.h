#ifndef KokkosCore_ProductBase_h
#define KokkosCore_ProductBase_h

#include <exception>
#include <memory>
#include <string>

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include "CUDACore/EventCache.h"
#include "CUDACore/SharedEventPtr.h"
#endif
#include "KokkosCore/ExecSpaceCache.h"
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
        ExecSpaceSpecific() : space_(getExecSpaceCache<ExecSpace>().get()) {}
        ~ExecSpaceSpecific() override = default;

        std::unique_ptr<ExecSpaceSpecific> cloneShareExecSpace() const {
          return std::make_unique<ExecSpaceSpecific>(*this);
        }

        std::unique_ptr<ExecSpaceSpecific> cloneShareAll() const { return std::make_unique<ExecSpaceSpecific>(*this); }

        void recordEvent() {}
        void enqueueCallback(edm::WaitingTaskWithArenaHolder withArenaHolder) {
          execSpace().fence();
          auto holder = withArenaHolder.makeWaitingTaskHolderAndRelease();
          holder.doneWaiting(nullptr);
        }
        void synchronizeWith(ExecSpaceSpecific const& other) const { other.execSpace().fence(); }

        ExecSpace const& execSpace() const { return *space_; }

      private:
        std::shared_ptr<ExecSpace> space_;
      };

#ifdef KOKKOS_ENABLE_CUDA
      template <>
      class ExecSpaceSpecific<Kokkos::Cuda> : public ExecSpaceSpecificBase {
      public:
        ExecSpaceSpecific() : ExecSpaceSpecific(getExecSpaceCache<Kokkos::Cuda>().get()) {}
        explicit ExecSpaceSpecific(std::shared_ptr<Kokkos::Cuda> execSpace)
            : ExecSpaceSpecific(std::move(execSpace), cms::cuda::getEventCache().get()) {}
        explicit ExecSpaceSpecific(std::shared_ptr<Kokkos::Cuda> execSpace,
                                   cms::cuda::SharedEventPtr event)
            : space_(std::move(execSpace)), event_(std::move(event)) {}

        ~ExecSpaceSpecific() override = default;

        std::unique_ptr<ExecSpaceSpecific> cloneShareExecSpace() const {
          return std::make_unique<ExecSpaceSpecific>(space_);
        }

        std::unique_ptr<ExecSpaceSpecific> cloneShareAll() const {
          return std::make_unique<ExecSpaceSpecific>(space_, event_);
        }

        void recordEvent() {
          // Intentionally not checking the return value to avoid throwing
          // exceptions. If this call would fail, we should get failures
          // elsewhere as well.
          cudaEventRecord(event_.get(), space_->cuda_stream());
        }

        void enqueueCallback(edm::WaitingTaskWithArenaHolder holder);

        void synchronizeWith(ExecSpaceSpecific const& other);

        int device() const { return space_->cuda_device(); }
        Kokkos::Cuda const& execSpace() const { return *space_; }

      private:
        bool isAvailable() const;

        std::shared_ptr<Kokkos::Cuda> space_;
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
          throw std::runtime_error(std::string("Incompatible Execution space: has ") + typeid(sp).name() + ", but " +
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
