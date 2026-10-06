"""Modules as namespaces that many threads can share.

On the free-threaded interpreter, `module.name` goes through the generic
attribute lookup of the module object every time, and every lookup updates the
reference count of the value it returns.  When many threads use the same names
-- `np.where`, `np.searchsorted` -- those reference counts become a point of
contention between the threads, and between the sockets of the machine: with
190 threads the reconstruction spent a quarter of its CPU time there.  A class
attribute lookup is specialised by the interpreter and avoids most of that, so
the modules here use numpy, and each other, through a class that holds the
names of the module:

    from namespaces import np
    np.where(...)

The class is a plain class on purpose: a metaclass with a __getattr__ to fall
back on the module, for names that are not in it, makes every lookup slower
than the module's.  So only the names the module holds when it is wrapped are
available -- numpy's lazily loaded submodules, such as np.linalg or np.random,
are not, and need to be imported explicitly.
"""

import numpy


def namespace(module):
    """The public names of `module`, as the attributes of a class."""
    names = {name: value for name, value in vars(module).items() if not name.startswith("__")}
    return type(module.__name__, (), names)


np = namespace(numpy)
