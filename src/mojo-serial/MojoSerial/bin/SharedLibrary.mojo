from std.ffi import OwnedDLHandle, RTLD

from MojoSerial.Framework.Wrapper import ErasedPointer, erase


# C++: edmplugin::SharedLibrary.
struct SharedLibrary(Movable):
    var _libraryHandle: OwnedDLHandle
    var _path: String

    def __init__(out self, path: String) raises:
        try:
            self._libraryHandle = OwnedDLHandle(path, RTLD.LAZY | RTLD.GLOBAL)
        except e:
            raise "RuntimeError: unable to load " + path + " because " + String(e)
        self._path = path

    # Like in C++ the library is never unloaded: the objects it created, and
    # the functions they point to, may outlive this struct. OwnedDLHandle would
    # close it, so leak the handle.
    def __deinit__(deinit self):
        _ = erase(self._libraryHandle^)

    def hasSymbol(self, name: String) -> Bool:
        return self._libraryHandle.check_symbol(name)

    # Call the function `name`, exported with the C ABI, that takes a pointer.
    def call(self, name: String, arg: ErasedPointer) raises:
        if not self.hasSymbol(name):
            raise "RuntimeError: no symbol " + name + " in " + self._path
        self._libraryHandle.get_function[NoneType](name)(arg)

    def path(self) -> String:
        return self._path
