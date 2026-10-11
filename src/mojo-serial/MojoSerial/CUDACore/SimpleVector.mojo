from MojoSerial.MojoBridge.DTypes import Typeable


struct SimpleVector[
    T: Copyable & Defaultable & Deinitable & Movable, DT: StaticString
](Copyable, Defaultable, Movable, Sized, Typeable):
    # C++ is {m_size, m_capacity, T* m_data} with the buffer allocated
    # separately on the device, so `construct` only adopts a caller's pointer.
    # There is no device here and every buffer was a sibling field of the
    # owner, so the vector owns its own. Capacity is len(m_data) (doc §19).
    var m_size: Int32
    var m_data: List[Self.T]

    @always_inline
    def __init__(out self):
        self.m_size = 0
        self.m_data = List[Self.T]()

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self.m_size = move.m_size
        self.m_data = move.m_data^

    @always_inline
    def __init__(out self, *, copy: Self):
        self.m_size = copy.m_size
        self.m_data = copy.m_data.copy()

    @always_inline
    def construct(mut self, var capacity: Int32):
        self.m_size = 0
        self.m_data = List[Self.T](length=Int(capacity), fill=Self.T())

    @always_inline
    def push_back_unsafe(mut self, ref element: Self.T) -> Int32:
        var previousSize = self.m_size
        self.m_size += 1

        if previousSize < self.capacity():
            self.m_data[Int(previousSize)] = element.copy()
            return previousSize
        else:
            self.m_size -= 1
            return -1

    @always_inline
    def back(ref self) -> ref [origin_of(self.m_data[0])] Self.T:
        if self.m_size > 0:
            return self.m_data[Int(self.m_size) - 1]
        return self.m_data[0]  # undefined behavior

    def push_back(mut self, ref element: Self.T) -> Int32:
        return self.push_back_unsafe(element)

    def extend(mut self, size: Int32 = 1) -> Int32:
        var previousSize = self.m_size
        self.m_size += size

        if previousSize < self.capacity():
            return previousSize
        else:
            self.m_size -= 1
            return -1

    def shrink(mut self, size: Int32 = 1) -> Int32:
        var previousSize = self.m_size
        self.m_size -= size

        if previousSize >= size:
            return previousSize - size
        else:
            self.m_size += size
            return -1

    @always_inline
    def empty(self) -> Bool:
        return self.m_size <= 0

    @always_inline
    def full(self) -> Bool:
        return self.m_size >= self.capacity()

    @always_inline
    def __getitem__(
        ref self, i: Int32
    ) -> ref [origin_of(self.m_data[0])] Self.T:
        return self.m_data[Int(i)]

    @always_inline
    def __setitem__(mut self, i: Int32, var val: Self.T):
        self.m_data[Int(i)] = val^

    @always_inline
    def reset(mut self):
        self.m_size = 0

    @always_inline
    def size(self) -> Int32:
        return self.m_size

    @always_inline
    def capacity(self) -> Int32:
        return Int32(self.m_data.__len__())

    @always_inline
    def data(ref self) -> Span[Self.T, origin_of(self.m_data)].Immutable:
        return Span(self.m_data)

    @always_inline
    def resize(mut self, size: Int32):
        self.m_size = size

    @always_inline
    def set_data(mut self, var data: List[Self.T]):
        self.m_data = data^

    @always_inline
    def __len__(self) -> Int:
        return Int(self.m_size)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SimpleVector[" + Self.DT + "]"


def make_SimpleVector[
    T: Copyable & Defaultable & Deinitable & Movable, DT: StaticString
](var capacity: Int32) -> SimpleVector[T, DT]:
    var ret = SimpleVector[T, DT]()
    ret.construct(capacity)
    return ret^


def make_SimpleVector[
    T: Copyable & Defaultable & Deinitable & Movable & Typeable, //
](var capacity: Int32) -> SimpleVector[T, T.dtype()]:
    var ret = SimpleVector[T, T.dtype()]()
    ret.construct(capacity)
    return ret^
