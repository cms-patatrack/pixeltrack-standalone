from std.memory import Layout

# A pointer to an object of a type that is known only to the code that created
# it, like a C++ `void*`. The plugins create their products, and the framework
# holds them, through these: the framework, built separately from the plugins
# (see bin/PluginManager.mojo), does not know the types of the products.
comptime ErasedPointer = OpaquePointer[MutUntrackedOrigin]


# Move `obj` to the heap, and forget its type.
def erase[T: Movable](var obj: T) -> ErasedPointer:
    var p = alloc[T](Layout[T](count=1)).unsafe_leak()
    p.unsafe_write(obj^)
    return p.unsafe_bitcast[NoneType]()


# Reinterpret an erased pointer as a pointer to `T`: the C++ `static_cast` from
# `void*`, or from the base class. The caller guarantees that the object is a T.
@always_inline
def unerase[T: AnyType](p: ErasedPointer) -> Pointer[T, MutUntrackedOrigin]:
    return p.unsafe_bitcast[T]()


# Destroy and free an object created by `erase[T]`; its address, taken from the
# code that knows T, is what stands for the virtual destructor.
def destroy_erased[T: Movable & Deinitable](p: ErasedPointer):
    var t = unerase[T](p)
    t.unsafe_deinit_pointee()
    t.unsafe_free()


# C++: `unique_ptr<WrapperBase>` holding a `Wrapper<T>` (Event.h), and the same
# for `ESWrapperBase` (EventSetup.h): owns an object of any type, which is
# destroyed through the virtual destructor.
struct Wrapper(Movable):
    var _obj: ErasedPointer
    var _destroy: def(ErasedPointer) thin -> None

    def __init__[T: Movable & Deinitable](out self, var obj: T):
        self._obj = erase(obj^)
        self._destroy = destroy_erased[T]

    def __deinit__(deinit self):
        self._destroy(self._obj)

    # C++: `static_cast<Wrapper<T> const&>(base).product()`. Unchecked, as in
    # C++: the typed tokens of the ProductRegistry, and the type names used as
    # keys of the EventSetup, guarantee that the object is a T.
    @always_inline
    def product[T: AnyType](ref self) -> ref [self] T:
        return (
            unerase[T](self._obj)
            .unsafe_mut_cast[origin_of(self).mut]()
            .unsafe_origin_cast[origin_of(self)]()[]
        )

    # The address of the object, for the code that knows its type.
    @always_inline
    def erased(self) -> ErasedPointer:
        return self._obj
