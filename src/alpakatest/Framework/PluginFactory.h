#ifndef PluginFactory_h
#define PluginFactory_h

#include <memory>
#include <string>
#include <unordered_map>

#include "Framework/Worker.h"

class ProductRegistry;

// Nothing here is thread safe
namespace edm {
  namespace PluginFactory {
    namespace impl {
      class MakerBase {
      public:
        virtual ~MakerBase() = default;

        virtual std::unique_ptr<Worker> create(ProductRegistry& reg) const = 0;
      };

      template <typename T>
      class Maker : public MakerBase {
      public:
        virtual std::unique_ptr<Worker> create(ProductRegistry& reg) const override {
          return std::make_unique<WorkerT<T>>(reg);
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

    std::unique_ptr<Worker> create(std::string const& name, ProductRegistry& reg);
  }  // namespace PluginFactory
}  // namespace edm

#define EDM_PLUGIN_SYM(x, y) EDM_PLUGIN_SYM2(x, y)
#define EDM_PLUGIN_SYM2(x, y) x##y

#define DEFINE_FWK_MODULE(type) static edm::PluginFactory::impl::Registrar<type> EDM_PLUGIN_SYM(maker, __LINE__)(#type);

#endif
