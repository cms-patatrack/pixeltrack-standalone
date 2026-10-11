@fieldwise_init
struct RBNode[
    K: ImplicitlyCopyable & Comparable & Deinitable,
    V: ImplicitlyCopyable & Deinitable,
](
    Copyable, ImplicitlyCopyable, Movable
):
    var key: Self.K
    var value: Self.V
    var parent: Int
    var left: Int
    var right: Int
    var red: Bool

    @always_inline
    def __init__(out self, key: Self.K, value: Self.V):
        self.key = key
        self.value = value
        self.parent = -1
        self.left = -1
        self.right = -1
        self.red = True


@fieldwise_init
struct OrderedMap[
    K: ImplicitlyCopyable & Comparable & Deinitable,
    V: ImplicitlyCopyable & Deinitable,
](Movable, Sized):
    # Index-based RB-tree: contiguous node storage for cache locality and low allocation overhead.
    var _nodes: List[RBNode[Self.K, Self.V]]
    var _root: Int
    var _size: Int

    @always_inline
    def __init__(out self):
        self._nodes = List[RBNode[Self.K, Self.V]]()
        self._root = -1
        self._size = 0

    @always_inline
    def __len__(self) -> Int:
        return self._size

    @always_inline
    def is_empty(self) -> Bool:
        return self._size == 0

    @always_inline
    def clear(mut self):
        self._nodes.clear()
        self._root = -1
        self._size = 0

    @always_inline
    def contains(self, key: Self.K) -> Bool:
        return self._find_index(key) != -1

    @always_inline
    def __contains__(self, key: Self.K) -> Bool:
        return self.contains(key)

    @always_inline
    def __getitem__(self, key: Self.K) raises -> Self.V:
        return self.get(key)

    @always_inline
    def __setitem__(mut self, key: Self.K, value: Self.V):
        _ = self.insert_or_assign(key, value)

    def get(self, key: Self.K) raises -> Self.V:
        var idx = self._find_index(key)
        if idx == -1:
            raise "KeyError: key not found"
        return self._nodes[idx].value

    # `out` is reserved for __init__ in 1.0; C++'s `bool try_get(K, V&)` maps to
    # a mutable borrow for the out-parameter.
    def try_get(self, key: Self.K, mut value: Self.V) -> Bool:
        var idx = self._find_index(key)
        if idx == -1:
            return False
        value = self._nodes[idx].value
        return True

    @always_inline
    def get_or(self, key: Self.K, default_value: Self.V) -> Self.V:
        var idx = self._find_index(key)
        if idx == -1:
            return default_value
        return self._nodes[idx].value

    # Returns True if a new node was inserted, False if an existing value was overwritten.
    def insert_or_assign(mut self, key: Self.K, value: Self.V) -> Bool:
        var parent = -1
        var cur = self._root

        while cur != -1:
            parent = cur
            var cur_key = self._nodes[cur].key
            if key < cur_key:
                cur = self._nodes[cur].left
            elif cur_key < key:
                cur = self._nodes[cur].right
            else:
                self._nodes[cur].value = value
                return False

        var new_idx = self._nodes.__len__()
        self._nodes.append(RBNode[Self.K, Self.V](key, value))
        self._nodes[new_idx].parent = parent

        if parent == -1:
            self._root = new_idx
        elif key < self._nodes[parent].key:
            self._nodes[parent].left = new_idx
        else:
            self._nodes[parent].right = new_idx

        self._fix_insert(new_idx)
        self._size += 1
        return True

    # Returns True if inserted, False if key already exists.
    def try_insert(mut self, key: Self.K, value: Self.V) -> Bool:
        var parent = -1
        var cur = self._root

        while cur != -1:
            parent = cur
            var cur_key = self._nodes[cur].key
            if key < cur_key:
                cur = self._nodes[cur].left
            elif cur_key < key:
                cur = self._nodes[cur].right
            else:
                return False

        var new_idx = self._nodes.__len__()
        self._nodes.append(RBNode[Self.K, Self.V](key, value))
        self._nodes[new_idx].parent = parent

        if parent == -1:
            self._root = new_idx
        elif key < self._nodes[parent].key:
            self._nodes[parent].left = new_idx
        else:
            self._nodes[parent].right = new_idx

        self._fix_insert(new_idx)
        self._size += 1
        return True

    # In-order traversal (sorted by key).
    def items(self) -> List[Tuple[Self.K, Self.V]]:
        var out = List[Tuple[Self.K, Self.V]]()
        var stack = List[Int]()
        var cur = self._root

        while cur != -1 or stack.__len__() > 0:
            while cur != -1:
                stack.append(cur)
                cur = self._nodes[cur].left

            cur = stack.pop()
            out.append((self._nodes[cur].key, self._nodes[cur].value))
            cur = self._nodes[cur].right

        return out^

    def keys(self) -> List[Self.K]:
        var out = List[Self.K]()
        var kv = self.items()
        for i in range(kv.__len__()):
            out.append(kv[i][0])
        return out^

    def values(self) -> List[Self.V]:
        var out = List[Self.V]()
        var kv = self.items()
        for i in range(kv.__len__()):
            out.append(kv[i][1])
        return out^

    @always_inline
    def _find_index(self, key: Self.K) -> Int:
        var cur = self._root
        while cur != -1:
            var cur_key = self._nodes[cur].key
            if key < cur_key:
                cur = self._nodes[cur].left
            elif cur_key < key:
                cur = self._nodes[cur].right
            else:
                return cur
        return -1

    @always_inline
    def _rotate_left(mut self, x: Int):
        var y = self._nodes[x].right
        var y_left = self._nodes[y].left
        self._nodes[x].right = y_left
        if y_left != -1:
            self._nodes[y_left].parent = x

        var x_parent = self._nodes[x].parent
        self._nodes[y].parent = x_parent

        if x_parent == -1:
            self._root = y
        elif x == self._nodes[x_parent].left:
            self._nodes[x_parent].left = y
        else:
            self._nodes[x_parent].right = y

        self._nodes[y].left = x
        self._nodes[x].parent = y

    @always_inline
    def _rotate_right(mut self, x: Int):
        var y = self._nodes[x].left
        var y_right = self._nodes[y].right
        self._nodes[x].left = y_right
        if y_right != -1:
            self._nodes[y_right].parent = x

        var x_parent = self._nodes[x].parent
        self._nodes[y].parent = x_parent

        if x_parent == -1:
            self._root = y
        elif x == self._nodes[x_parent].right:
            self._nodes[x_parent].right = y
        else:
            self._nodes[x_parent].left = y

        self._nodes[y].right = x
        self._nodes[x].parent = y

    def _fix_insert(mut self, start_idx: Int):
        var z = start_idx

        while z != self._root:
            var p = self._nodes[z].parent
            if p == -1 or not self._nodes[p].red:
                break

            var gp = self._nodes[p].parent
            if gp == -1:
                break

            if p == self._nodes[gp].left:
                var y = self._nodes[gp].right

                if y != -1 and self._nodes[y].red:
                    self._nodes[p].red = False
                    self._nodes[y].red = False
                    self._nodes[gp].red = True
                    z = gp
                else:
                    if z == self._nodes[p].right:
                        z = p
                        self._rotate_left(z)
                        p = self._nodes[z].parent
                        gp = self._nodes[p].parent

                    self._nodes[p].red = False
                    self._nodes[gp].red = True
                    self._rotate_right(gp)
            else:
                var y = self._nodes[gp].left

                if y != -1 and self._nodes[y].red:
                    self._nodes[p].red = False
                    self._nodes[y].red = False
                    self._nodes[gp].red = True
                    z = gp
                else:
                    if z == self._nodes[p].left:
                        z = p
                        self._rotate_right(z)
                        p = self._nodes[z].parent
                        gp = self._nodes[p].parent

                    self._nodes[p].red = False
                    self._nodes[gp].red = True
                    self._rotate_left(gp)

        if self._root != -1:
            self._nodes[self._root].red = False
