from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct VecArray[
    T: ImplicitlyCopyable & Movable & Deinitable,
    DT: StaticString,
    maxSize: Int,
](Copyable, Defaultable, Movable, Sized, Typeable):
    # 1.0 dropped `run_destructors`; InlineArray always destroys its elements.
    var m_data: InlineArray[Self.T, Self.maxSize]
    var m_size: Int32
    comptime ValueType = Self.T

    @always_inline
    def __init__(out self):
        self.m_data = InlineArray[Self.T, Self.maxSize](uninitialized=True)
        self.m_size = 0

    @always_inline
    def push_back_unsafe(mut self, ref element: Self.T) -> Int32:
        var previousSize = self.m_size
        self.m_size += 1

        if previousSize < Int32(Self.maxSize):
            self.m_data[previousSize] = element
            return previousSize
        else:
            self.m_size -= 1
            return -1

    @always_inline
    def back(self) -> ref [self.m_data] Self.T:
        if self.m_size > 0:
            return self.m_data[self.m_size - 1]
        else:
            return self.m_data[0]  # undefined behavior

    def push_back(mut self, ref element: Self.T) -> Int32:
        return self.push_back_unsafe(element)

    @always_inline
    def pop_back(mut self) -> Self.T:
        if self.m_size > 0:
            var previousSize = self.m_size
            self.m_size -= 1
            return self.m_data[previousSize - 1]
        else:
            return self.m_data[0]  # undefined behavior

    # C++ exposes `T* begin()` / `T* end()`; the pair delimits [0, m_size), which
    # is one Span here. Neither had a caller, so this replaces both.
    @always_inline
    def span(ref self) -> Span[Self.T, origin_of(self.m_data)]:
        return Span(self.m_data)[: Int(self.m_size)]

    @always_inline
    def __getitem__(ref self, i: Int32) -> ref [self.m_data] Self.T:
        return self.m_data[i]

    @always_inline
    def reset(mut self):
        self.m_size = 0

    @always_inline
    @staticmethod
    def capacity(self) -> Int32:
        return Int32(Self.maxSize)

    # C++: `T const* data() const`
    @always_inline
    def data(self) -> Span[Self.T, origin_of(self.m_data)].Immutable:
        return Span(self.m_data)

    @always_inline
    def resize(mut self, var size: Int32):
        self.m_size = size

    @always_inline
    def empty(self) -> Bool:
        return self.m_size == 0

    @always_inline
    def full(self) -> Bool:
        return self.m_size == Int32(Self.maxSize)

    @always_inline
    def __len__(self) -> Int:
        return Int(self.m_size)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "VecArray[" + Self.DT + ", " + String(Self.maxSize) + "]"
