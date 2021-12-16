#ifndef HeterogeneousCore_AlpakaCore_Event_h
#define HeterogeneousCore_AlpakaCore_Event_h

#include "AlpakaCore/alpakaConfigFwd.h"
#include "AlpakaCore/EDContext.h"
#include "AlpakaCore/ProductMetadata.h"
#include "Framework/Event.h"
#include "Framework/ProductEDGetToken.h"
#include "Framework/ProductEDPutToken.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace impl {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
    template <typename T>
    struct GetHelper {
      static auto const& get(edm::Event const& event, EDContext& ctx, edm::EDGetTokenT<T> const& token) {
        return event.get(token);
      }
    };

    template <typename T>
    struct EmplaceHelper {
      template <typename... Args>
      static void emplace(edm::Event& event, EDContext& ctx, edm::EDPutTokenT<T> const& token, Args&&... args) {
        event.emplace(token, std::forward<Args>(args)...);
      }
    };
#else // asynchronous backends
    template <typename T>
    struct GetHelper {
      static auto const& get(edm::Event const& event, EDContext& ctx, edm::EDGetTokenT<T> const& token) {
        auto const& product = event.get(edm::ProductEDGetToken<T>::toProductToken(token));
        return ctx.get(product);
      }
    };

    template <typename T>
    struct EmplaceHelper {
      template <typename... Args>
      static void emplace(edm::Event& event, EDContext& ctx, edm::EDPutTokenT<T> const& token, Args&&... args) {
        event.emplace(edm::ProductEDPutToken<T>::toProductToken(token), ctx.metadataPtr(), std::forward<Args>(args)...);
      }
    };
#endif

    // Specializations for host products are the same
    template <typename T>
    struct GetHelper<edm::Host<T>> {
      static auto const& get(edm::Event const& event, EDContext& ctx, edm::EDGetTokenT<edm::Host<T>> const& token) {
        return event.get(edm::ProductEDGetToken<T>::toToken(token));
      }
    };

    template <typename T>
    struct EmplaceHelper<edm::Host<T>> {
      template <typename... Args>
        static void emplace(edm::Event& event, EDContext& ctx, edm::EDPutTokenT<edm::Host<T>> const& token, Args&&... args) {
        event.emplace(edm::ProductEDPutToken<T>::toToken(token), ctx.metadataPtr(), std::forward<Args>(args)...);
      }
    };
  }

  class Event {
  public:
    Event(edm::Event const& ev, EDContext& ctx): constEvent_(ev), ctx_(ctx) {}
    Event(edm::Event& ev, EDContext& ctx): constEvent_(ev), event_(&ev), ctx_(ctx) {}

    auto streamID() const { return constEvent_.streamID(); }
    auto eventID() const { return constEvent_.eventID(); }

    template <typename T>
    auto const& get(edm::EDGetTokenT<T> const& token) const {
      return impl::GetHelper<T>::get(constEvent_, ctx_, token);
    }

    template <typename T, typename... Args>
    void emplace(edm::EDPutTokenT<T> const& token, Args&&... args) {
      impl::EmplaceHelper<T>::emplace(*event_, ctx_, token, std::forward<Args>(args)...);
    }

  private:
    edm::Event const& constEvent_;
    edm::Event* event_ = nullptr;
    EDContext& ctx_;
  };
}

#endif
