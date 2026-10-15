#include "PluginFactory.h"

#include <stdexcept>

namespace edm {
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

    std::unique_ptr<Worker> create(std::string const& name, ProductRegistry& reg) {
      return impl::getGlobalRegistry().get(name)->create(reg);
    }
  }  // namespace PluginFactory
}  // namespace edm
