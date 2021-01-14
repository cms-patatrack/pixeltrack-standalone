#ifndef KokkosCore_ContextState_h
#define KokkosCore_ContextState_h

#include "KokkosCore/ProductBase.h"

namespace cms {
  namespace kokkos {
    template <typename T>
    class ScopedContextAcquire;
    template <typename T>
    class SCopedContextProduce;

    /**
     * The purpose of this class is to deliver the device and CUDA stream
     * information from ExternalWork's acquire() to producer() via a
     * member/StreamCache variable.
     */
    template <typename ExecSpace>
    class ContextState {
    public:
      ContextState() = default;
      ~ContextState() = default;

      ContextState(const ContextState&) = delete;
      ContextState& operator=(const ContextState&) = delete;
      ContextState(ContextState&&) = delete;
      ContextState& operator=(ContextState&& other) = delete;

    private:
      friend class ScopedContextAcquire<ExecSpace>;
      friend class ScopedContextProduce<ExecSpace>;

      void set(std::unique_ptr<impl::ExecSpaceSpecific<ExecSpace>> specific) { specific_ = std::move(specific); }

      std::unique_ptr<impl::ExecSpaceSpecific<ExecSpace>> release() { return std::move(specific_); }

      //void throwIfStream() const;
      //void throwIfNoStream() const;

      std::unique_ptr<impl::ExecSpaceSpecific<ExecSpace>> specific_;
    };
  }  // namespace kokkos
}  // namespace cms

#endif
