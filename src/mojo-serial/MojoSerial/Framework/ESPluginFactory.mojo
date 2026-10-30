from std.os import abort
from std.pathlib import Path

from MojoSerial.Framework.ESProducer import ESProducer
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.Wrapper import (
    ErasedPointer,
    destroy_erased,
    erase,
    unerase,
)

# See Framework/PluginFactory.mojo for the vtables.


# C++: unique_ptr<ESProducer>, pointing to the producer of a plugin.
struct ESProducerPtr(Movable):
    var _producer: ErasedPointer
    var _produce: def(ErasedPointer, mut EventSetup) thin -> None
    var _destroy: def(ErasedPointer) thin -> None

    def __init__(
        out self,
        producer: ErasedPointer,
        produce: def(ErasedPointer, mut EventSetup) thin -> None,
        destroy: def(ErasedPointer) thin -> None,
    ):
        self._producer = producer
        self._produce = produce
        self._destroy = destroy

    def __deinit__(deinit self):
        self._destroy(self._producer)

    def produce(mut self, mut eventSetup: EventSetup):
        self._produce(self._producer, eventSetup)


# C++: ESPluginFactory::impl::Maker<T>, behind a MakerBase.
@fieldwise_init
struct ESMaker(Copyable, ImplicitlyCopyable, Movable):
    var _create: def(Path) thin -> ErasedPointer
    var _produce: def(ErasedPointer, mut EventSetup) thin -> None
    var _destroy: def(ErasedPointer) thin -> None

    @staticmethod
    def of[T: ESProducer & Deinitable]() -> Self:
        return Self(_create[T], _produce[T], destroy_erased[T])

    def create(self, datadir: Path) -> ESProducerPtr:
        return ESProducerPtr(self._create(datadir), self._produce, self._destroy)


def _create[T: ESProducer](datadir: Path) -> ErasedPointer:
    return erase(T(datadir.copy()))


def _produce[T: ESProducer](p: ErasedPointer, mut eventSetup: EventSetup):
    unerase[T](p)[].produce(eventSetup)


# C++: edm::ESPluginFactory::impl::Registry, and ESPluginFactory::create(). C++
# keeps it in a static variable; here the PluginManager owns it.
struct ESRegistry(Movable):
    var _pluginRegistry: Dict[String, ESMaker]

    def __init__(out self):
        self._pluginRegistry = Dict[String, ESMaker]()

    def add(mut self, name: String, maker: ESMaker) raises:
        if name in self._pluginRegistry:
            raise "LogicError: Plugin " + name + " is already registered"
        self._pluginRegistry[name] = maker

    def get(self, name: String) raises -> ESMaker:
        if name not in self._pluginRegistry:
            raise "LogicError: Plugin " + name + " is not registered"
        return self._pluginRegistry[name]

    def create(self, name: String, datadir: Path) raises -> ESProducerPtr:
        return self.get(name).create(datadir)


# C++: DEFINE_FWK_EVENTSETUP_MODULE(T). Called by the function that the plugin
# library exports for T (see bin/PluginManager.mojo), with the ESRegistry of the
# framework.
def define_fwk_eventsetup_module[
    T: ESProducer & Deinitable
](registry: ErasedPointer):
    try:
        unerase[ESRegistry](registry)[].add(T.dtype(), ESMaker.of[T]())
    except e:
        abort(String(e))
