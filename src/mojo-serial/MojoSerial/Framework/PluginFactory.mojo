from std.memory import ArcPointer
from std.os import abort

from MojoSerial.Framework.EDProducer import EDProducer, EDProducerWithGlobals
from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.Wrapper import (
    ErasedPointer,
    Wrapper,
    destroy_erased,
    erase,
    unerase,
)

# Mojo 1.0 has no virtual functions and no trait objects: the C++ vtables of
# Worker and MakerBase are spelled out as structs of function pointers, filled
# by the plugin library with the instances of generic functions for the type of
# the producer. The framework calls them without knowing that type.


# C++: the virtual functions of Worker, implemented by WorkerT<T>.
@fieldwise_init
struct WorkerVTable(Copyable, ImplicitlyCopyable, Movable):
    var produce: def(ErasedPointer, mut Event, EventSetup) thin -> None
    var endJob: def(ErasedPointer) thin raises -> None
    var destroy: def(ErasedPointer) thin -> None

    @staticmethod
    def of[T: EDProducer & Deinitable]() -> Self:
        return Self(_produce[T], _endJob[T], destroy_erased[T])


def _produce[
    T: EDProducer
](p: ErasedPointer, mut event: Event, eventSetup: EventSetup):
    unerase[T](p)[].produce(event, eventSetup)


def _endJob[T: EDProducer](p: ErasedPointer) raises:
    unerase[T](p)[].endJob()


# C++: unique_ptr<Worker>, pointing to a WorkerT<T> that holds the producer.
struct Worker(Movable):
    var _producer: ErasedPointer
    var _vtable: WorkerVTable

    def __init__(out self, producer: ErasedPointer, vtable: WorkerVTable):
        self._producer = producer
        self._vtable = vtable

    def __deinit__(deinit self):
        self._vtable.destroy(self._producer)

    @always_inline
    def produce(mut self, mut event: Event, ref eventSetup: EventSetup):
        self._vtable.produce(self._producer, event, eventSetup)

    def endJob(mut self) raises:
        self._vtable.endJob(self._producer)


# C++: Maker<T>, behind a MakerBase. `_globals`, when set, holds an
# ArcPointer[T.Globals]: the state that the C++ plugin keeps in static variables.
struct Maker(Movable):
    var _create: def(mut ProductRegistry, Optional[ErasedPointer]) thin -> ErasedPointer
    var _vtable: WorkerVTable
    var _globals: Optional[Wrapper]

    @staticmethod
    def of[T: EDProducer & Deinitable]() -> Self:
        return Self(_create[T], WorkerVTable.of[T](), None)

    @staticmethod
    def of[T: EDProducerWithGlobals & Deinitable](streams: Int) -> Self:
        return Self(
            _create_with_globals[T],
            WorkerVTable.of[T](),
            Wrapper(ArcPointer(T.makeGlobals(streams))),
        )

    def __init__(
        out self,
        create: def(mut ProductRegistry, Optional[ErasedPointer]) thin -> ErasedPointer,
        vtable: WorkerVTable,
        var globals: Optional[Wrapper],
    ):
        self._create = create
        self._vtable = vtable
        self._globals = globals^

    def create(self, mut reg: ProductRegistry) -> Worker:
        var globals: Optional[ErasedPointer] = None
        if self._globals:
            globals = self._globals.value().erased()
        return Worker(self._create(reg, globals), self._vtable)


def _create[
    T: EDProducer
](mut reg: ProductRegistry, globals: Optional[ErasedPointer]) -> ErasedPointer:
    return erase(T(reg))


def _create_with_globals[
    T: EDProducerWithGlobals
](mut reg: ProductRegistry, globals: Optional[ErasedPointer]) -> ErasedPointer:
    return erase(T(reg, unerase[ArcPointer[T.Globals]](globals.value())[]))


# C++: edm::PluginFactory::impl::Registry, and PluginFactory::create(). C++ keeps
# it in a static variable; here the PluginManager owns it.
struct Registry(Movable):
    # the number of streams that share the globals of the plugins
    var _streams: Int
    var _pluginRegistry: Dict[String, Maker]

    def __init__(out self, streams: Int):
        self._streams = streams
        self._pluginRegistry = Dict[String, Maker]()

    def add(mut self, name: String, var maker: Maker) raises:
        if name in self._pluginRegistry:
            raise "LogicError: Plugin " + name + " is already registered"
        self._pluginRegistry[name] = maker^

    def get(self, name: String) raises -> ref [self._pluginRegistry[name]] Maker:
        if name not in self._pluginRegistry:
            raise "LogicError: Plugin " + name + " is not registered"
        return self._pluginRegistry[name]

    def streams(self) -> Int:
        return self._streams

    def create(self, name: String, mut reg: ProductRegistry) raises -> Worker:
        return self.get(name).create(reg)


# C++: DEFINE_FWK_MODULE(T). Called by the function that the plugin library
# exports for T (see bin/PluginManager.mojo), with the Registry of the framework.
def define_fwk_module[T: EDProducer & Deinitable](registry: ErasedPointer):
    try:
        unerase[Registry](registry)[].add(T.dtype(), Maker.of[T]())
    except e:
        abort(String(e))


# DEFINE_FWK_MODULE for an EDProducerWithGlobals: creates its globals.
def define_fwk_module_with_globals[
    T: EDProducerWithGlobals & Deinitable
](registry: ErasedPointer):
    try:
        ref r = unerase[Registry](registry)[]
        r.add(T.dtype(), Maker.of[T](r.streams()))
    except e:
        abort(String(e))
