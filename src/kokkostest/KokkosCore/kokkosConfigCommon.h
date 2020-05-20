#ifndef KokkosCore_kokkosConfigCommon_h
#define KokkosCore_kokkosConfigCommon_h

#include <memory>
#include <unordered_map>

// This header needs to be #included in a file that may not be
// compiled with nvcc.
namespace kokkos_common {
  class InitializeScopeGuardBase {
  public:
    InitializeScopeGuardBase();
    virtual ~InitializeScopeGuardBase() = 0;
  };

  namespace PluginFactory {
    namespace impl {
      class MakerBase {
      public:
        virtual ~MakerBase() = default;

        virtual std::unique_ptr<InitializeScopeGuardBase> create() const = 0;
      };

      template <typename T>
      class Maker : public MakerBase {
      public:
        virtual std::unique_ptr<InitializeScopeGuardBase> create() const override { return std::make_unique<T>(); };
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

    std::unique_ptr<InitializeScopeGuardBase> create(std::string const& name);
  }  // namespace PluginFactory

}  // namespace kokkos_common

#define EDM_KOKKOSINIT_PLUGIN_SYM(x, y) EDM_KOKKOSINIT_PLUGIN_SYM2(x, y)
#define EDM_KOKKOSINIT_PLUGIN_SYM2(x, y) x##y

#define DEFINE_FWK_KOKKOSINIT_CLASS(type) \
  static kokkos_common::PluginFactory::impl::Registrar<type> EDM_KOKKOSINIT_PLUGIN_SYM(maker, __LINE__)(#type);

#endif
