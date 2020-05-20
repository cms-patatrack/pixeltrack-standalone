#include "KokkosCore/kokkosConfigCommon.h"

#include <stdexcept>

namespace kokkos_common {
  InitializeScopeGuardBase::InitializeScopeGuardBase() = default;
  InitializeScopeGuardBase::~InitializeScopeGuardBase() = default;

  namespace PluginFactory {
    namespace impl {
      void Registry::add(std::string const& name, std::unique_ptr<MakerBase> maker) {
        auto found = pluginRegistry_.find(name);
        if (found != pluginRegistry_.end()) {
          throw std::logic_error("Plugin " + name + " is already registered");
        }
        pluginRegistry_.emplace(name, std::move(maker));
      }

      MakerBase const* Registry::get(std::string const& name) {
        auto found = pluginRegistry_.find(name);
        if (found == pluginRegistry_.end()) {
          throw std::logic_error("Plugin " + name + " is not registered");
        }
        return found->second.get();
      }

      Registry& getGlobalRegistry() {
        static Registry reg;
        return reg;
      }
    };  // namespace impl

    std::unique_ptr<InitializeScopeGuardBase> create(std::string const& name) {
      return impl::getGlobalRegistry().get(name)->create();
    }
  }  // namespace PluginFactory
}  // namespace kokkos_common
