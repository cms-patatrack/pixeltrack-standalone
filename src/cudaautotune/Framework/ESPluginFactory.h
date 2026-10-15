#ifndef PluginFactory_h
#define PluginFactory_h

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include "Framework/ESProducer.h"

class ProductRegistry;

// Nothing here is thread safe
namespace edm {
  namespace ESPluginFactory {
    namespace impl {
      class MakerBase {
      public:
        virtual ~MakerBase() = default;

        virtual std::unique_ptr<ESProducer> create(std::filesystem::path const& datadir) const = 0;
      };

      template <typename T>
      class Maker : public MakerBase {
      public:
        virtual std::unique_ptr<ESProducer> create(std::filesystem::path const& datadir) const override {
          return std::make_unique<T>(datadir);
        };
      };

      class Registry {
      public:
        void add(std::string const& name, std::unique_ptr<MakerBase> maker);
        MakerBase const* get(std::string const& name);

      private:
        std::unordered_map<std::string, std::unique_ptr<MakerBase>> pluginRegistry_;
      };

      Registry& getGlobalRegistry();

      template <typename T>
      class Registrar {
      public:
        Registrar(std::string const& name) { getGlobalRegistry().add(name, std::make_unique<Maker<T>>()); }
      };
    }  // namespace impl

    std::unique_ptr<ESProducer> create(std::string const& name, std::filesystem::path const& datadir);
  }  // namespace ESPluginFactory
}  // namespace edm

#define EDM_ES_PLUGIN_SYM(x, y) EDM_ES_PLUGIN_SYM2(x, y)
#define EDM_ES_PLUGIN_SYM2(x, y) x##y

#define DEFINE_FWK_EVENTSETUP_MODULE(type) \
  static edm::ESPluginFactory::impl::Registrar<type> EDM_ES_PLUGIN_SYM(maker, __LINE__)(#type);

#endif
