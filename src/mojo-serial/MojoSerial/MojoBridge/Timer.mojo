from std.time import perf_counter_ns


struct Timer(Copyable, Defaultable, Movable, TrivialRegisterPassable):
    var _start: Int
    var _time: Int

    @always_inline
    def __init__(out self):
        self._start = 0
        self._time = 0

    @always_inline
    def __init__(out self, var start: Int):
        self._start = start
        self._time = 0

    @always_inline
    def start(mut self):
        self._start = perf_counter_ns()

    @always_inline
    def finish(mut self):
        self._time += perf_counter_ns() - self._start

    @always_inline
    def get(self) -> Int:
        return self._time

    @always_inline
    def finalize(self, var name: String):
        print(
            "[" + name + "] completed in ",
            self._time // (10**6),
            "ms",
            sep="",
        )

    @always_inline
    def __str__(self) -> String:
        return (
            "Timer(" + String(self._start) + ", " + String(self._time) + ")"
        )


struct TimerManager(Defaultable, Movable, Sized):
    # Dict and List are already heap-backed handles, so the OwnedPointer only
    # added an indirection -- and `ptr[]` yields an rvalue, which is why every
    # mutating call through it was rejected. Plain fields plus `mut self`.
    var _storage: Dict[String, Timer]
    # a stack
    var _cur: List[String]

    @always_inline
    def __init__(out self):
        self._storage = Dict[String, Timer]()
        self._cur = List[String]()

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._storage = move._storage^
        self._cur = move._cur^

    @always_inline
    def __enter__(mut self):
        if not self.empty():
            var key = self.top()
            if key not in self._storage:
                self._storage[key] = Timer()
            try:
                self._storage[key].start()
            except e:
                print(e)

    @always_inline
    def __exit__(mut self):
        try:
            var key = self.top()
            self._storage[key].finish()
        except e:
            print(e)
        self.pop()

    @always_inline
    def configure(mut self, var name: String):
        self._cur.append(name^)

    @always_inline
    def start(mut self, var name: String = ""):
        if name:
            self.configure(name^)
        self.__enter__()

    @always_inline
    def stop(mut self):
        self.__exit__()

    @always_inline
    def clear(mut self):
        self._storage.clear()
        self._cur.clear()

    @always_inline
    def empty(self) -> Bool:
        return self._cur.__len__() == 0

    # Returns a copy rather than a borrow: every caller uses it as a Dict key
    # while mutating _storage, which a live borrow of _cur would not allow.
    @always_inline
    def top(self) -> String:
        return self._cur[self._cur.__len__() - 1].copy()

    @always_inline
    def pop(mut self):
        _ = self._cur.pop()

    @always_inline
    def __len__(self) -> Int:
        return self._storage.__len__()

    @always_inline
    def finalize(mut self):
        try:
            while not self.empty():
                self.stop()
            # keys are copied out first: iterating the Dict borrows it, which
            # would conflict with reaching each value mutably below
            var names = List[String]()
            for k in self._storage.keys():
                names.append(k.copy())
            for name in names:
                self._storage[name].finalize(name.copy())
        except e:
            print(e)
