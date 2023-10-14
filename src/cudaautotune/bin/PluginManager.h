#ifndef PluginManager_h
#define PluginManager_h

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "SharedLibrary.h"

namespace edmplugin {
  class PluginManager {
  public:
    PluginManager();

    SharedLibrary const& load(std::string const& pluginName);

  private:
    std::unordered_map<std::string, std::string> pluginToLibrary_;

    std::recursive_mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<SharedLibrary>> loadedPlugins_;
  };
}  // namespace edmplugin

#endif
