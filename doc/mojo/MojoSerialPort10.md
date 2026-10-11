# MojoSerial: porting `src/mojo-serial` from Mojo 25.5 to Mojo 1.0

Working notes and reference for the 25.5.0 → 1.0.0 migration on branch
`mojo-serial-1.0-port`. Everything here was verified against the 1.0 compiler on
this source or on an isolated probe — nothing is recalled from documentation.

> **STATE:** the tree is at **29 errors across 13 files** — 6 of those are the
> spurious `main() in packages`, so **23 real errors across 11 files**. Both
> self-referential views are gone (`TrackingRecHit2DSOAView` §11, `ParamsOnGPU`
> §13) and the whole RecHits chain — `PixelCPEforGPU`, `PixelCPEFast`,
> `TrackingRecHit2DHeterogeneous`, `PixelRecHits`, `GPUPixelRecHits`,
> `SiPixelRecHitCUDA` — is at zero, as is the `gpuVertexFinder` cluster.
> `OwnedPointer` fields are down from 39 to 1 (§12). The plugin factories are
> deleted in favour of a closed `Variant` set in `bin/Plugins.mojo`, and
> `StreamSchedule` no longer holds `_source`/`_eventSetup` — **§17 is still owed
> for that change.** Two tests now *execute*: `GPUClusteringTest` matches the
> C++ reference byte-for-byte, and a vertex-finder driver passes on hand-checked
> input.
>
> The raw-decode chain — `FEDRawData`, `FEDHeader`, `FEDTrailer`,
> `ErrorChecker`, `WordFedAppender`, `fillHitsModuleStart` — is at zero (§18).
> What is left is the `SimpleVector` cluster (12), the two test files (9) and
> two strays.

---

## 1. Toolchain

`pixi.toml` now pins:

```toml
mojo = "==1.0.0"
# provides `layout` (LayoutTensor); not part of the stdlib, so imported unprefixed
max  = "==26.5.0"
```

- `max 26.5.0` installs cleanly alongside `mojo 1.0.0` and supplies `layout.mojoc`.
- **`layout` is NOT under `std.`** — keep `from layout import Layout, LayoutTensor`.
- Version path from 25.5: 25.6 → 25.7 → 26.1 → 26.2 → 1.0 (five releases of drift).

Typecheck the whole package without needing a `main`:

```
pixi run mojo precompile MojoSerial -I . -o /tmp/x.mojoc
```

`precompile` rejects `main()` inside a package, so `bin/main.mojo` and the test
files each contribute one spurious "not supported within packages" error. Discount 6.

---

## 2. Language changes (Phase A — mechanical)

Applied by script, 2,665 edits across 117 of 127 files.

| Change | 25.5 | 1.0 | Sites |
|---|---|---|---|
| `fn` removed | `fn f()` | `def f()` | 1341 |
| …including function *types* | `fn (Int) capturing -> None` | `def (Int) capturing -> None` | 13 |
| compile-time binding | `alias K = 4` | `comptime K = 4` | 794 |
| compile-time control flow | `@parameter` + newline + `for`/`if` | `comptime for` / `comptime if` | 287 |
| stdlib moved under one package | `from sys import …` | `from std.sys import …` | 115 |
| size intrinsics snake_case | `sizeof`, `alignof` | `size_of`, `align_of` | 48 |
| trivial types become a trait | `@register_passable("trivial")` | `struct S(TrivialRegisterPassable)` | 30 |
| decorator removed outright | `@nonmaterializable(NoneType)` | *(deleted)* | 16 |
| origin intrinsic de-dundered | `__origin_of(x)` | `origin_of(x)` | 11 |
| type intrinsic de-dundered | `__type_of(x)` | `type_of(x)` | 1 |
| destructor renamed | `def __del__(var self)` | `def __deinit__(var self)` | 5 |
| `constrained` left the prelude | *(builtin)* | `from std.builtin import constrained` | 7 files |

**Semantics preserved.** Two properties verified before trusting the `fn`→`def`
sweep, both of which would have silently changed behaviour tree-wide:

- A plain 1.0 `def` **does not raise**. Calling a `raises` function from one fails
  with *"cannot call function that may raise in a context that cannot raise."*
- A plain 1.0 `def` argument **defaults to a read/borrow**, not by-value. Tested with
  an instrumented copy constructor; no copy fires.

`@parameter` on a **closure** is unchanged — only `@parameter for` / `@parameter if`
become `comptime`. The rewrite must distinguish them.

### Phase B — compiler fix-its

1.0 rejects unqualified access to a struct's own parameters and prints the exact
replacement. Scraping the diagnostics and applying them fixed **318** sites across
22 files in a single pass (a second pass found none).

```
error: unqualified access to struct parameter 'T'; use 'Self.T' instead
    comptime Scalar = Scalar[T]
                             ^
                             Self.T
```

---

## 3. Stdlib relocation map

Everything moved under `std.` except where noted.

| 25.5 | 1.0 |
|---|---|
| `sys`, `time`, `pathlib`, `memory`, `math`, `collections`, `bit`, `hashlib`, `utils`, `os`, `random`, `builtin`, `algorithm` | prefix with `std.` |
| `sys.ffi` | **`std.ffi`** (top level) |
| `os.atomic` | **`std.atomic`** (top level) |
| `compile.reflection` | **`std.reflection`** |
| `Span` | **`std.collections`** |
| `constrained` | `std.builtin` |
| `layout`, `LayoutTensor` | **unprefixed** — ships in MAX, not the stdlib |

Gone with no drop-in replacement:

- `algorithm.functional.parallelize` — not in stdlib *or* MAX. Use
  `std.runtime.asyncrt.TaskGroup` / `create_task`. One call site (`main.mojo:188`).
- `compile.reflection.get_type_name` — `std.reflection` has `get_linkage_name` only.
- `DType.sizeof()` — the method is removed; use the free function `size_of[DType.uint64]()`.
- `__str__` on scalars/DType — use `String(x)`.
- `Stringable`, `Representable` — no longer declared.

The stdlib also renamed every unsafe operation to carry an `unsafe_` prefix:
`bitcast`→`unsafe_bitcast`, `memset`→`unsafe_memset`,
`take_pointee`→`unsafe_take_pointee`, `steal_data`→`unsafe_take_allocation`,
pointer `+`/`+=`→`unsafe_offset`. Together with the `UnsafePointer` deprecation this
is a deliberate campaign to make unsafety lexically visible at every call site.

---

## 4. The pointer / origin model

**`UnsafePointer` is deprecated in 1.0.** The compiler says so directly:
`warning: 'UnsafePointer' is deprecated, use 'Pointer' instead`. Its `origin`
parameter is also now mandatory, so every pointer declaration has to be touched for
the 1.0 port regardless — the migration and the pointer refactor are the same job.

### Origins are compile-time only

| type | bytes |
|---|---|
| `Pointer[UInt16, origin_of(l)]` | 8 — same as a raw pointer |
| `Span` immutable | 16 — pointer + length |
| `Span[mut=True, …]` | 16 — *identical* |

Origins are type-level parameters in the `[]` list, erased before codegen. Borrow
checking costs nothing at runtime.

### The spellings that matter

```mojo
Span[T, _]                 # read-only view, origin unbound       (parameters)
Span[mut=True, T, _]       # writable view, no origin parameter   (parameters)
Span[T, origin_of(self.x_d[])].Immutable   # explicitly const     (returns)
Pointer[T, origin_of(self.bins)]           # single object / iterator
ref [self.field] T                         # reference to one object
```

Non-obvious facts, each verified:

- **`mut` on the binding does nothing for element writes.** `mut xx: Span[T, _]`
  makes `xx` itself reassignable; element mutability lives in the **origin**.
- **`Span[T, _]` is a real read-only contract.** Passing a *mutable* span to such a
  parameter still rejects writes — `_` binds `mut=False`, it does not inherit the
  caller's mutability. And the call still type-checks (mut → imm conversion).
- **`Span[mut=True, T, _]` keeps alias detection.** `_` unbinds the origin in the
  *signature*; the compiler still infers the concrete origin per call site:
  `error: aliasing values passed mutably to 'a' argument and passed mutably to 'b'`.
- **`.Immutable` requires a bound `mut`.** `Span[T, _].Immutable` is invalid
  (*"cannot access comptime member 'Immutable' with unbound parameter 'Span.mut'"*),
  so const-ness is spelled differently in parameters vs returns.
- **`Pointer` supports the whole C++ iterator idiom** — `+`, `+=`, `<`, `[]`, and
  `Int(ptr)` for addresses. Converting `begin()`/`end()` from `UnsafePointer` to
  `Pointer` needed no call-site changes at all.
- `MutAnyOrigin`, `MutUntrackedOrigin`, `ImmUntrackedOrigin` all exist.

### `Origin[mut]` is gone — declare a bare `Origin`

Pre-1.0, a struct parametric over mutability declared a companion `Bool` and used
it to index `Origin`:

```mojo
struct _MatIterator[
    mat_mutability: Bool, //,          # inferred, never passed
    ...
    mat_origin: Origin[mat_mutability],
```

1.0 makes that unwritable — `error: unexpected parameter` — because its `Origin`
puts *both* parameters behind the infer-only marker:

```
struct Origin[mut: Bool, _mlir_origin: LITOrigin[mut], //]
```

The mutability does not disappear, it moves inside. Declare the parameter as bare
`Origin` and read the flag back off it:

```mojo
struct _MatIterator[W: DType, rows: Int, colns: Int, mat_origin: Origin, ...]:
    var src: Pointer[Self.mat_type, Self.mat_origin]
    # ... String(Self.mat_origin.mut) -> "True"
```

Two consequences. The companion `Bool` parameter disappears from the signature
(and so from `dtype()` strings). And **inside a struct body every struct parameter
must be `Self.`-qualified** — `Matrix[W, rows, colns]` becomes
`Matrix[Self.W, Self.rows, Self.colns]`. This is by far the most common mechanical
error in the vendored-stdlib files; `Matrix.mojo` alone had ~25 of them.

Fixing just the two iterator signatures (`_VecIterator`, `_MatIterator`) cleared
31 downstream errors on its own.

### Borrow granularity — the rule that shapes every signature

Mojo checks borrows against the **signature**, not the body. A method taking
`ref self` declares a borrow of the *entire* struct, and is held to that even when
the body touches one field.

| form | borrows | several at once? |
|---|---|---|
| `soa.xx()` — `ref self` | the whole struct, exclusively | **no** |
| `soa.c_xx()` — `self` | the whole struct, shared | yes, and alongside field writes |
| `Span(soa.xx_d[])` | just the path `soa.xx_d` | yes — fields are disjoint |

Consequence: an SoA **cannot** expose usable mutable whole-array accessors.
`RawToDigi_kernel` takes six digis arrays in one call; six exclusive borrows of
`digis_d` cannot coexist. Field borrows can. This matches the pre-existing FIXME at
`GPUPixelRecHits.mojo:26` — *"not using views (passing a gazzilion of array
pointers) seems to produce the fastest code"*.

---

## 5. Conversion recipes

### SoA accessors

Three families, mirroring C++'s const/non-const pair plus the old view's element access:

```mojo
def xx(self, i: Int) -> UInt16:                                    # value; no borrow held
    return self.xx_d[][i]

def xx(ref self) -> Span[UInt16, origin_of(self.xx_d[])]:          # exclusive, one at a time
    return Span(self.xx_d[])

def c_xx(self) -> Span[UInt16, origin_of(self.xx_d[])].Immutable:  # shared, many at once
    return Span(self.xx_d[])
```

### Kernel signatures

```mojo
def RawToDigi_kernel(
    cablingMap: SiPixelFedCablingMapGPU,     # single object -> read borrow
    modToUnp: Span[UChar, _],                # read-only array
    xx: Span[mut=True, UInt16, _],           # writable array
    mut err: SimpleVector[...],              # container being pushed into
)
```

Call sites pass disjoint field borrows for the writable ones and keep `c_*` for the
read-only ones — this preserves the read/write intent and satisfies the checker:

```mojo
GPUClustering.findClus(
    self.digis_d.c_moduleInd(),              # read  — shared borrow
    self.clusters_d.c_moduleStart(),         # read
    Span(self.clusters_d.clusInModule_d[]),  # write — field borrow
    Span(self.digis_d.clus_d[]),             # write
)
```

### ES product accessors

```mojo
def getCPUProduct(self) -> ref [self.cablingMapHost] SiPixelFedCablingMapGPU:
    return self.cablingMapHost
```

Caller must bind with `ref`, not `var`. The types enforce this: these products are
`Movable` but **not** `Copyable`, so a `var` binding fails to compile rather than
silently deep-copying a cabling map.

### Binary search

Prefer Span + index over pointer pairs — it deletes the arithmetic entirely:

```mojo
def upper_bound[T: DType, //](s: Span[Scalar[T], _], var value: Scalar[T]) -> Int
```

```mojo
# before
var off = upper_bound(offsets, offsets + nh + 1, i)
var ih: Int32 = ((Int(off) - Int(offsets)) // size_of[UInt32]()) - 1
# after
var off = upper_bound(offsets[0 : Int(nh) + 1], UInt32(i))
var ih = Int32(off) - 1
```

### Self-referential views

A struct owning buffers *and* a view into them cannot name the origin — a field
cannot reference a sibling field's origin (`error: use of unknown declaration 'self'`).
Two outcomes:

1. **Delete the view** where it exists only for CUDA kernel marshalling. Both
   `DeviceConstView`s went this way; their hand-written `__moveinit__`, which existed
   solely to rebuild the view after a move, went with them.
2. **Untracked origin** where the view is genuinely needed across structs:
   `Span[T, ImmUntrackedOrigin]`, built with `rebind`. Lifetime is unchecked, but the
   element type and length survive — strictly better than a raw pointer.

---

## 6. Numeric strictness

1.0 **removed the implicit `UInt` → `Int` conversion** (it is not value-preserving).
This is the root of a large error class:

- `List(length=…)` takes `Int` — `List[T](length=UInt(n), fill=0)` fails.
- `range(a, b)` needs both `Int` — `range(UInt32, Int32)` has no overload.
- `min`/`max` will not mix `Int` with `Int32`.
- C++ relies on implicit narrowing that must now be spelled: `hist.fill(y[i], i - firstPixel)`
  becomes `hist.fill(y[i], UInt16(i - Int(firstPixel)))`.

**Fixing one conversion creates the next.** Changing `range(first, numElements)` to
`range(Int(first), Int(numElements))` made `i` an `Int`, which then broke
`min(msize, i)`. This category cannot be swept mechanically — each site needs the C++
checked to learn which type is authoritative.

---

## 6b. Stdlib API changes — the `MojoBridge` Eigen shim

`MojoBridge/Vector.mojo` and `Matrix.mojo` are forks of the stdlib `SIMD`, so they
absorb every stdlib API break at once. They sit at the bottom of the dependency
graph (`EigenSoA`, `TrajectoryStateSoA`, `BrokenLine`, `RiemannFit`,
`HelixFitOnGPU` all build on them), so fixing them cleared **137** errors
tree-wide — 313 → 176.

| 25.5 | 1.0 | Note |
| --- | --- | --- |
| `a < b` on `SIMD` width > 1 | `a.lt(b)` | also `.le .gt .ge .eq .ne` |
| `Origin[mut]` | bare `Origin`, flag via `.mut` | see §4 |
| `InlineArray(x)` splat | `InlineArray(fill=x)` | |
| `Matrix[T, *_]` | `Matrix[T, ...]` | unbound parameters |
| `String.write(x)` | `String(x)` | |
| `T is target` on `DType` | `T == target` | `DType` has no `__is__` |
| `__origin_of(x)` | `origin_of(x)` | |
| `__type_of(x)` | `type_of(x)` | |
| `OwnedPointer.take()` | `.into_inner()` | consuming |
| `@doc_private`, `AnyTrivialRegType` | *removed* | |

**SIMD comparison operators are now `Scalar`-only.** The constraint message is
explicit: *"Strict inequality is only defined for `Scalar`s; did you mean to use
`SIMD.lt(...)`?"* This is a real trap — `a < b` on a wide SIMD was elementwise in
25.5 and is a hard error in 1.0, so there is no silent-behaviour-change risk, but
every masked comparison in the shim had to be rewritten.

**`InlineArray` is not `ImplicitlyCopyable`,** so a struct holding one can no longer
synthesize its copy constructor:

```
error: cannot synthesize implicit copy constructor because field '_data' has
non-implicitly-copyable type 'Array[Vector[T, colns], rows]'
```

Dropping `ImplicitlyCopyable` from `Matrix` would force `.copy()` at hundreds of
by-value call sites in `BrokenLine`/`RiemannFit`. Verified alternative: **write the
copy constructor by hand** and the conformance still holds.

```mojo
def __init__(out self, *, copy: Self):
    self._data = Self._DC(uninitialized=True)
    comptime for i in range(Self.rows):
        self._data[i] = copy._data[i]
```

(Same trait break hit `QualityCuts.chi2Coeff`; there the fix was
`InlineArray[Float32, 4]` → `SIMD[DType.float32, 4]`, which is implicitly copyable
and indexes identically.)

**`OwnedPointer[T]` now requires `T: Deinitable`** — `field '_inner' has
non-'Deinitable' type 'OwnedPointer[T]'`. Add the bound to the wrapper's parameter.

**`LayoutTensor` has no `UnsafePointer` constructor.** Its 1.0 overloads take a
`Span` or a tracked `Pointer`. This turned a comment into a compiler guarantee:

```mojo
# was: -> LayoutTensor[..., MutAnyOrigin]  built from buf.unsafe_ptr()
#      with a comment saying "buf must outlive the returned tensor"
def to_layout_tensor[T: DType, rows: Int, cols: Int](
    m: Matrix[T, rows, cols], mut buf: InlineArray[Scalar[T], rows * cols]
) -> LayoutTensor[mut=True, T, Layout.col_major(rows, cols), origin_of(buf)]:
    ...
    return LayoutTensor[...  origin_of(buf)](Span(buf))
```

An argument's origin can be named in the return type, so the buffer's lifetime is
now checked instead of documented.

**Dropped as dead:** `DevicePassable` (declared only by `Vector`, zero callers, and
its `_to_device_type` signature changed to an encoder form — the serial backend
never launches a kernel); the `__mlir_type.index` constructors on `Vector` and
`Matrix` (unreachable from nameable Mojo code); `HeterogeneousSoA`'s `Traits` /
`CPUTraits` (`@deprecated`, no conformers, the port already collapsed the policy to
`OwnedPointer`); and `TypeableOwnedPointer.take()`/`unsafe_ptr()` — every caller
wanted the pointee, so `__getitem__` covers them all and one more `UnsafePointer`
leaves the tree.

---

## 7. Pitfalls and defects found

**Silent-corruption near-misses in the mechanical sweep** — both produced
plausible-looking source that the error count would not have caught:

- the `@register_passable` rule dropped a comma on the two structs whose conformance
  list spans multiple lines;
- the `@parameter for` rule swallowed a newline, leaving `comptime` orphaned on its
  own line.

Dry-run every rule against a copy and *read the diff*, not just the error count. A
blanket `sizeof`→`size_of` also corrupted a C++ cross-reference comment
(`HelixFitOnGPU.mojo:137`), which reverted; it was the only comment hit in 2,665 edits.

**A third instance, later and self-inflicted.** Converting `Rfit.printIt` from
`UnsafePointer[M]` to a borrow meant stripping `UnsafePointer(to=…)` from 56 call
sites in `RiemannFit.mojo`, done as one `replace_all` per argument name. Three
matches were not `printIt` calls at all:

```mojo
print("Address of p2D: ", UnsafePointer(to=p2D))   # C++: printf("... %p\n", &p2D)
```

The rule rewrote them to `print("Address of p2D: ", p2D)` — turning *print the
address* into *print the matrix*. Caught only because `M2xN` does not conform to
`Writable`; with a printable type it would have compiled and printed the wrong
thing forever. Restored as `Pointer(to=…)`: still an address, still tracked.

The lesson is narrower than "read the diff": **a textual rule keyed on the
argument cannot see the enclosing call.** When converting a function's parameter
convention, drive the edit from the *callee* name, not the argument spelling.

**Byte/element confusion.** C++ `h.end(b) - h.begin(b)` is pointer subtraction on
`IndexType*` and yields an *element* count. The port wrote
`Int(h.end(b)) - Int(h.begin(b))`, a *byte* difference — 4× too large with the default
`IndexType = uint32`, and the following `debug_assert(tot == rtot)` would fire. Five
sites in `test/HistoContainerTest.mojo`; every other pointer difference in the tree
divided correctly. Fixed with `h.size(b)`, which computes `off[b+1] - off[b]` directly
and removes the pointer arithmetic rather than correcting it.

**Compile-time tables indexed at runtime.** `ilog2`'s lookup tables were `alias` in
25.5 and indexed by a runtime loop variable — legal then, not in 1.0. Making them
`var` would lose their constness; the fix is `comptime for` plus extracting each
element as a comptime scalar, so they stay constants and never materialize:

```mojo
comptime for i in range(4, -1, -1):
    comptime bi = b[i]
    if v & bi: ...
```

**Dead debug code inflates the error count.** `comptime CPP_DUMP = False` in
`BrokenLine.mojo` gates 53 lines holding **34** of that file's 80 errors. Three
sibling flags (`RIEMANN_DEBUG`, `BROKENLINE_DEBUG`, `BL_DUMP_HITS`) are also
hardcoded false.

**A descending `range` from an unsigned start is silently empty.** Probed:
`range(UInt32(5), 0, -1)` iterates **zero** times; `range(Int(5), 0, -1)`
iterates five. No warning, no error. C++ `for (auto i = MaxNumModules; i > 0;
i--)` is a `uint32_t` countdown, so the two `last module is …` loops in
`GPUClusteringTest` were dead code — the C++ reference prints that line five
times and the port printed it zero. Caught only by diffing against the C++
binary. Fix is `range(Int(...), 0, -1)`; a sweep found only one other descending
range in the tree and it starts from an `Int` literal.

**`comptime assert False` at a function tail is evaluated unconditionally.**
Used as an "unreachable" sentinel at the end of `signed_to_unsigned`'s
`comptime if/elif` chain — where 1.0 removed `DType.invalid` — it rejected
*every* `T`, including the `uint16` the chain explicitly handles, with
*"constraint failed: signed_to_unsigned requires an integral DType"* at each
instantiation site. The `return`s above it do not make it unreachable. Probed:
the same assert inside an untaken `else:` branch is **not** evaluated (only an
"unreachable code" warning), so the sentinel must live in an `else`, never at
the tail. Note this differs from the `comptime if` behaviour below, which
typechecks the branch it does not take — typechecking a branch and evaluating
its asserts are not the same thing.

---

## 8. Reading the error count

**Fixing an upstream file raises the total**, because the compiler then reaches code
it previously could not parse. Clearing `HistoContainer` took that file 28 → 0 while
moving the total 706 → 751: `CAHitNtupletGeneratorKernels` went 50 → 80 and three
files began reporting errors for the first time. The same happened when `layout`
started resolving (793 → 802).

Use **files-at-zero**, not the total, as the progress metric.

---

## 9. Status

- 802 → **86** errors
- `UnsafePointer` 598 → **101**; `Span` 2 → **139**
- self-referential views 4 → **0** — none left in the port
- `OwnedPointer` fields 39 → **1** (only `TypeableOwnedPointer._inner`, the
  Event-product mechanism; see §12)
- of 125 `.mojo` files, **27 still carry at least one error**

**Two tests now execute, not merely compile.** `test/GPUClusteringTest` is
byte-for-byte identical to the C++ reference (`g++ -I. test/cpuClustering_t.cc`)
across all five iterations, and a `VertexFinder_t`-style driver over the vertex
finder passes on a hand-checked 3-vertex input. Running them is the only reason
the two bugs in §7 (`range(UInt32, …, -1)`, and the tail `comptime assert`) were
found — neither shows up in the error count.

Count errors with `grep "error:"`, not by counting lines that merely start with
a filename — most files now carry more *warnings* than errors, and mixing them
overstates the remaining work.
- 313 warnings remain, most of them the `UnsafePointer` deprecation — i.e. the
  remaining worklist restated

Fully converted: `SiPixelClustersSoA`, `SiPixelDigisSoA`, `HistoContainer`,
`CUDAStdAlgorithm`, `GPUCalibPixel`, `SiPixelFedCablingMapGPUWrapper`,
`SiPixelGainForHLTonGPU`, `SiPixelGainCalibrationForHLTGPU`, `GPUPixelRecHits`,
`SiPixelRecHitCUDA`. Plus `RawToDigi_kernel`, `countModules`, `findClus`,
`clusterChargeCut`, and the whole `cablingMap` thread.

Also fully converted and pointer-free: the **`gpuVertexFinder` cluster** —
`gpuVertexFinder`, `gpuVertexFinderImpl`, `gpuClusterTracksByDensity` /
`DBSCAN` / `Iterative`, `gpuFitVertices`, `gpuSortByPt2`, `gpuSplitVertices`,
plus `ZVertexSoA`. C++ passes `ZVertices* __restrict__` / `WorkSpace*
__restrict__` — single objects, asserted non-null, never offset — so all of it
became `mut data: ZVertices, mut ws: WorkSpace` and the `ref data = pdata[]`
prologues disappeared. Mojo's guarantee that two `mut` borrows cannot alias is
exactly what `__restrict__` was asserting by hand. `test/GPUClusteringTest` and
`CUDACore/PrefixScan` are done too.

## 10. Remaining work

Grouped by dependency vertical rather than by size, since the fix order is forced.

**The fit chain (~48)** — the last unported piece of the reconstruction spine.
Bottom-up: `SymmetricEigen` (7), `EigenSoA` (8), `TrajectoryStateSoA` (11) →
`RiemannFit` (7), `BrokenLine` (1) → `RiemannFitOnGPU` (1), `BrokenLineFitOnGPU` (1)
→ `HelixFitOnGPU` (10) → `CAHitNtupletGeneratorOnGPU` (2). All of it now rests on
the finished `Matrix`/`Vector` shim. `TrajectoryStateSoA`'s 11 errors are *all* the
`LayoutTensor` missing-origin error solved in §6b.

**The framework spine (~46)** — needed before the binary links and runs an event:
`StreamSchedule` (9), `main` (8), `Event` / `EventSetup` / `PluginFactory` /
`ESPluginFactory` / `EventProcessor` (5 each), `Source` (4), `ProductRegistry` (3).

**Container shims (~27)** — `OrderedMultiSet` (11), `Timer` (10), `OrderedMap` (6).
Independent of everything else; `Timer`'s errors are all one shape (mutating a
`Dict`/`List` through a non-`mut` accessor).

**Leaf files (~40)** — `SiPixelRawToClusterGPUKernel` (5), `FEDRawData` (5),
`SiPixelDigisSoA` / `FEDTrailer` / `FEDHeader` / `SimpleVector` (4 each), and a long
tail of 1–3.

**Tests (~10)** — `HistoContainerTest` (7), plus singletons. Lowest priority.

Cross-cutting, not a file: `Int`/`UInt32` conversions, not mechanically sweepable
(§6).

---

## 11. DONE — removing `TrackingRecHit2DSOAView`

**Complete.** 722 → 675 errors across the whole change. No view type remains in
the port; `TrackingRecHit2DSOAView.mojo` is deleted.

### Decision taken

Three options were weighed; the chosen one was **delete the view and unpack the
stores**, accepting layout divergence from `src/serial` in exchange for no views
and no untracked origins anywhere.

Rejected alternatives, for the record:

- *Convert the view in place* (37 pointer fields → `Span[mut=True, T, MutUntrackedOrigin]`).
  Keeps the C++ layout exactly, but leaves one untracked-origin view standing.
  This shape was verified working, including the punned column.
- *Delete the view, accessors on the owner, keep packed stores.* Punned-column
  **writes** need a pointer-level bitcast to yield a mutable reference, so raw
  pointers come back at ~4 sites.

### Done

- `TrackingRecHit2DHeterogeneous.mojo` rewritten: 13 typed column buffers, the
  view's accessors folded on, `m_view` gone.
- `TrackingRecHit2DSOAView.mojo` **deleted**. `Hist` (nested in the struct in
  C++) now lives in `TrackingRecHit2DHeterogeneous.mojo`; the one consumer that
  imported it, `gpuPixelDoubletsAlgo.mojo`, imports it from there.
- All nine consumers repointed. Four were pure alias retargets
  (`HitsOnGPU`/`HitsView`/`Hits` → `TrackingRecHit2DHeterogeneous`); the rest
  had `UnsafePointer[View]` parameters, all of which became plain borrows.
- `GPUPixelRecHits.getHits` converted whole: five pointer parameters
  (`cpeParams`, `bs`, `pdigis`, `pclusters`, `phits`) became borrows, with
  `mut hits` the only mutable one. This file carried all three deleted views
  (digis, clusters, hits) and is now at **0 errors**, as is
  `SiPixelRecHitCUDA.mojo`.

### Column mapping (old packed → new unpacked)

Old wire-up used two closures local to the constructor,
`get32(i) = m_store32.unsafe_ptr() + i * nHits` and the same over `m_store16` —
"start of column `i` at stride `nHits`". They existed only to wire the view's
pointers and have no replacement.

| old | new field |
|---|---|
| `get32(0..7)` | `m_xl_d`, `m_yl_d`, `m_xerr_d`, `m_yerr_d`, `m_xg_d`, `m_yg_d`, `m_zg_d`, `m_rg_d` — `List[Float]` |
| `get32(8).bitcast[Int32]()` | `m_charge_d: List[Int32]` |
| `get32(9)`, the `+11` tail | `m_hitsLayerStart_d: List[UInt32]`, length `numberOfLayers + 1` |
| `get16(0).bitcast[Int16]()` | `m_iphi_d: List[Int16]` |
| `get16(1)` | `m_detInd_d: List[UInt16]` |
| `get16(2).bitcast[Int16]()` | `m_xsize_d: List[Int16]` |
| `get16(3).bitcast[Int16]()` | `m_ysize_d: List[Int16]` |

Five `bitcast`s eliminated; the C++ `static_assert(sizeof(uint32_t) == sizeof(float))`
that licensed the punning is no longer load-bearing.

Fields that collapsed — the old design carried each supporting object twice, once
on the owner and once on the view: `m_hist`, `m_iphi`, `m_hitsLayerStart`,
`m_averageGeometry`. Also gone: `m_view`, `comptime n16`/`n32`, and the
hand-written `__moveinit__` (now synthesizable).

Two forwarded (non-owned) fields changed representation rather than being unpacked:

- `m_cpeParams: UnsafePointer[ParamsOnGPU]` → **`ParamsOnGPU` by value.**
  Measured 32 bytes — it is four pointers and nothing else, so copying it copies
  addresses, not geometry. Note its own four fields are still `UnsafePointer`;
  `PixelCPEforGPU.mojo` is a separate job and currently does not compile.
- `m_hitsModuleStart: UnsafePointer[UInt32]` → **copied into an owned
  `List[UInt32]`** (~8 KB/event). Equivalent because `clusters_d` is not mutated
  after this point, but it is a copy that did not exist before.

### Cost, corrected

An earlier note in this session said "14 allocations instead of 2" — that was
wrong, conflating column buffers with total allocations. Actual per-event
`OwnedPointer` count:

| | old | new |
|---|---|---|
| column storage | 2 | 13 |
| Hist / AverageGeometry | 2 | 2 |
| the view itself | 1 | 0 |
| hitsLayerStart | 0 (lived in the `m_store32` tail) | 1 |
| hitsModuleStart | 0 (borrowed) | 1 |
| **total** | **5** | **17** |

This is a steeper cost than was quoted when the option was chosen. It is a
one-allocation-per-column consequence of unpacking and is not recoverable
without repacking the stores; noted here so the tradeoff stays visible.

### `ref` returns: the origin must name the exact expression

Found while fixing the 13 folded accessors. Mojo 1.0 accepts **no widening** of
a returned reference's origin — it must name the exact access path:

```mojo
def x(ref self, i: Int) -> ref [self.m_xl_d[]] Float:      # ✗ names the List
def x(ref self, i: Int) -> ref [self] Float:               # ✗ widening to self
def x(ref self, i: Int) -> ref [self.m_xl_d[][i]] Float:   # ✓ names the element
```

The error reads `cannot return reference with incompatible origin:
'origin_of(self.m_xl_d["value"]["element"])' vs 'origin_of(self.m_xl_d["value"])'`
— `["element"]` is the tell that the declared origin stopped one level short.
Widening a plain field (`ref [self]` returning `self.p`) is rejected the same
way; `ref [self.p]` is the accepted form.

### Verification

Per-file counts after the change: `GPUPixelRecHits` and `SiPixelRecHitCUDA` at
**0**; `TrackingRecHit2DHeterogeneous` at **2**, both cascades from
`ParamsOnGPU` in the still-un-ported `PixelCPEforGPU.mojo` (it is not `Movable`
and its field origin degrades to `MutUnsafeAnyOrigin`). Every residual error in
the other seven consumers is the generic
`UnsafePointer[T]`-without-origin category, pre-existing and unrelated.

**`PixelCPEforGPU.mojo` is therefore the next file** — it is the only thing
still holding the hit SoA back.

---

## 12. Codegen — measured, not assumed

All figures below are from `mojo build --emit asm`, default `-O3`, x86-64.
Probes live in the scratchpad (`asm/layout.mojo`, `asm/spans.mojo`,
`asm/thirteen.mojo`).

### Benchmark with `-D ASSERT=none` or you measure bounds checks

With assertions on (the default) the same loop is **207 lines** of assembly
instead of **31** — `String::write_to`, `_debug_assert_msg` and `_printf` cold
paths for every `List.__getitem__`. Any codegen comparison without
`-D ASSERT=none` is measuring the assertion machinery, not the code.

### Per-element accessors defeat the backend

An `@always_inline` accessor returning `ref [self.m_xl_d[][i]]` inlines fine,
but the `OwnedPointer -> List -> data` indirection is **reloaded on every
iteration**. A store through any column may alias the pointer cells holding the
other columns' base addresses, so LLVM cannot hoist them.

Loop reading 3 columns and writing a 4th:

| form | loop body | pointer reloads / iter |
|---|---|---|
| accessors on the unpacked struct | 14 instr | **4** |
| packed store (the C++ `m_store32` layout) | 12 instr | 1 |
| Spans bound before the loop | **9 instr** | **0** |
| Spans as parameters | **9 instr** | **0** |

At the real column count (13, `store_accessors` vs `store_spans`) the gap
widens rather than closing — register pressure was the worry, and it did not
materialise:

| form | loop body | pointer reloads / iter |
|---|---|---|
| accessors | 33 instr | **13** |
| Spans bound before the loop | **18 instr** | **0** |

All 13 base addresses stay in registers; only two loop-invariant scalars spill.

**Consequence: binding columns to Spans once, before the loop, beats even the
packed C++ layout** — packed still reloads its single base pointer and computes
column offsets. The unpacking decision (§11) therefore costs nothing at the
access site, provided hot loops bind Spans instead of calling per-element
accessors. Applied to `GPUPixelRecHits.getHits`; the 50 iterator sites in §10
are the same pattern.

This also vindicates the upstream C++ FIXME in `getHits` ("not using views —
passing a gazzilion of array pointers — seems to produce the fastest code, but
it is harder to maintain"). A Span *is* that array pointer, with the bounds
carried along, so the maintainability objection does not transfer.

Note the accessors must stay on the struct regardless — they are the API for
scalar, non-loop access. The Span binding is a hot-loop technique, not a
replacement.

### `Int(<UInt32 comptime>)` does not fold in parameter position

`Phase1PixelTopology` hit this and it is easy to misdiagnose. A list-literal
`InlineArray` whose length is `Int(Self.numberOfLayers) + 1` fails with a bare
*"no matching function in initialization"*; the real cause is buried in the
candidate notes:

```
return type 'Array[UInt32, Int(11)]' parameter 'length' value 'Int(11)'
doesn't match expected value '(SIMD(UInt32(10)) + Int(1))'
```

The literal deduces `length = Int(11)`; the annotation stays the *symbolic*
`SIMD(UInt32(10)) + Int(1)` and the two never unify. Binding it to an
intermediate `comptime … : Int` does not help (it becomes `Int((add SIMD(...),
1))`), and wrapping in `comptime (…)` is rejected — *"expression is already
evaluated at compile time"*. It is folded; it just does not **normalize** for
parameter matching.

The fix is to make the constant a plain `Int` so no SIMD appears in the
expression. Note this only bites where a list literal must match a deduced size:
`InlineArray[UInt8, Int(<UInt32>)](fill=0)` is fine, which is why
`PixelCPEforGPU` never saw it.

### Higher-order parametric functions no longer bind

`_map_to_array[…, func: fn[Scalar[I]]() -> Scalar[R]]()` cannot be called or
passed in 1.0 (*"parameter 'func' has `def[Scalar[I], /]() -> Scalar[R]` type,
but value has type `def findLayerFromCompact[detId: UInt32]() thin -> UInt8`"*).
It had a single caller, so it was deleted and the loop inlined into a
purpose-built `_build_layer_table()`. If a future case has many callers this
will need a real answer.

### Runtime-indexing a `comptime` table: `materialize`

A `comptime InlineArray` indexed by a **runtime** value fails with *"cannot
materialize comptime value of type `Array[Float32, N]` to runtime because it is
not `ImplicitlyCopyable`"*. The table does **not** have to become a `var` — the
compiler's own fix-it is `materialize`:

```mojo
comptime tbl: InlineArray[Float, 3] = [0.00115, 0.00120, 0.00088]
...
materialize[tbl]()[Int(sx)] if Int(sx) < comptime (len(tbl)) else tbl_def
```

`materialize` is a builtin — do **not** import it (`std.builtin` has no such
name). Two further notes: `len()` on a comptime table materialises it too, so it
needs `comptime (len(tbl))`; and a `comptime` **SIMD** vector indexes by a
runtime value with no ceremony at all, so it is the lighter choice for a table
that is only ever read.

Applied to the six error tables in `errorFromSize`/`errorFromDB`.

**`materialize` rebuilds the table on every call — prefer SIMD.** This is not a
style preference. `FEDNumbering.inRange` was ported as
`materialize[_in]()[i]` over a `comptime List[Bool]` of 4097 entries, which
means a heap allocation and a 4097-element fill *per call* to a hot predicate.
Three things were probed to settle the alternatives:

- **Module scope does not help.** A module-level `comptime InlineArray` read at
  a runtime index gives the same materialize error as a struct member.
- **There is no global `var`.** *"global variables are not supported; move this
  into a function body or use 'comptime' to declare a constant"* — so C++'s
  `static bool in_[MAXFEDID+1]`, built once at static init, has no direct
  equivalent.
- **`SIMD[DType.bool, N]` does not scale.** `N = 8192` fails to instantiate.

What works is a `comptime` SIMD of **`UInt64` used as a bitmask**: 4097 flags in
128 lanes, indexed at a runtime index with no materialize. `initIn` keeps its
original shape — 27 `comptime for` loops — with `_in[i] = True` becoming
`_in[i // 64] |= UInt64(1) << UInt64(i % 64)`, and the lookup is
`(_in[i // 64] >> UInt64(i % 64)) & 1 == 1`. Note the shift RHS needs an
explicit `UInt64(...)`; a bare `i % 64` is a `_SequentialRange.Element` and is
rejected.

Verified exhaustively against an independent oracle over all 4097 ids (1147 in
range, 0 mismatches) — the compile passes either way, so only the oracle catches
a dropped range. An intermediate attempt that replaced the table with a
hand-written chain of range tests silently dropped 3 of the 27 ranges, because
the ranges were extracted with a regex that missed the ones split across lines.

### `mut` on a mid-sized struct is copy-in/copy-out, not a reference

Measured with `@no_inline` and a single-field write, so the numbers are the
per-call overhead of the calling convention alone:

| struct size | instrs/call | behaviour |
|---|---|---|
| 16 B | 3 | by-reference (register-passed) |
| 88 B | 13 | **copy in and back out** |
| 128 B | 17 | **copy in and back out** |
| 256 B | 25 | **copy in and back out** |
| 1024 B | 2 | by-reference |

Inside the band the callee reads the whole struct and writes the whole struct
back, even to touch one field. `ref [o] x: T` compiles **byte-identically** to
`mut x: T` — there is no borrow convention that avoids it. Only
`Pointer[T, o]` is genuinely by-reference there (7 instrs vs 16 for an 88-byte
`Counters`, 4 memory ops vs 11).

Practical rule: `mut`/`ref` are free for small structs and for large ones
(SoA containers, `HitContainer`, `TkSoA`). For structs of roughly a few dozen
to a few hundred bytes **on a hot path**, prefer `Pointer[T, o]` as a
*parameter*. `Counters` (11 × UInt64 = 88 B) is in the band but every call site
is behind `if m_params.doStats`, so it keeps `mut`.

### A read-only borrow is already by-reference — `ref` adds nothing

The band above is about **mutable** parameters. For a *read-only* parameter the
default borrow is passed by reference at every size, and `ref [o] m: T` is
byte-identical to it. Measured on a 288-byte struct (the shape of
`Rfit.Matrix6d`), both `@no_inline`:

```asm
take_borrow(Big):              take_ref(Big%):
  vmovsd  (%rdi), %xmm0          vmovsd  (%rdi), %xmm0
  vaddsd  280(%rdi), %xmm0       vaddsd  280(%rdi), %xmm0
  retq                           retq
```

The call sites are `movq %rbx, %rdi; callq …` for both — same address, no
temporary. Only the mangled name differs (`Big` vs `Big%`).

So `ref` is **never** a performance choice for a read-only parameter. What it
buys is a **named origin**, needed only when a reference *escapes*: into the
return type, or into a stored field. That is the rule:

| situation | spelling |
|---|---|
| reads it, returns nothing derived from it | `m: T` |
| returns a view/reference into it | `ref self` + `origin_of(self.field)` |
| writes through it, small or huge | `mut m: T` |
| writes through it, in the size band, hot | `Pointer[T, o]` |

Examples in this port: `printIt(m: M)` and `Scatter_cov_line(…)` take plain
borrows (nothing escapes); `EigenSoA.data(ref self) -> Span[…,
origin_of(self._data)]` and `MatrixSoA.__getitem__(ref self, …) ->
LayoutTensor[…, origin_of(self._data)]` need `ref` because the returned view
points into `self`. Note both name **`self._data`**, not `self` — the Span is
built from the field, and a whole-struct origin will not unify with it
(*"cannot be converted from `Span[Scalar[T], origin_of(_mlir_origin._data)]` to
`Span[Scalar[T], origin]`"*).

### A struct cannot hold a `ref` field

`var b: ref [o] Big` is rejected — *"'ref' patterns are only valid on the left
side of an assignment"*. So when a struct needs to reference something it does
not own there are only three options:

1. `Pointer[T, o]` — tracked, but forces an origin parameter onto the struct,
   which then spreads to every use of that type and makes construction awkward.
2. Own it — `OwnedPointer` (heap) or a plain inline field.
3. **Don't store it** — take `mut` as a parameter and let the caller own it.

Option 3 is usually right, and it is how `counters_` was removed: C++ held
`Counters*` pointing at the generator's counters, but the kernels object is
rebuilt per event, so an owned field would have accumulated nothing. Passing
`mut counters: Counters` keeps accumulation on the generator, which is what
C++ actually does.

### `OwnedPointer` is almost never worth it — SUPERSEDED

The original table below justified boxing the big containers. **It no longer
holds**: once `HistoContainer`, `ZVertexSoA` and `WorkSpace` got heap-backed
columns (see the compile-time-bomb section above), every one of those types
collapsed to a handful of bytes, and the boxes became pure overhead.

| held type | size before | size now | verdict |
|---|---|---|---|
| `SimpleVector` (2×Int32 + ptr) | 16 B | 16 B | inline field |
| `TupleMultiplicity` | ~48 KB | **56 B** | inline field |
| `HitToTuple` | ~384 KB | **56 B** | inline field |
| `HitContainer` | ~338 KB | **56 B** | inline field |
| `Hist` (rechit) | 101 KB | **56 B** | inline field |
| `ZVertexSoA` | 216 KB | **176 B** | inline field |
| `WorkSpace` | ~600 KB | **160 B** | inline field |

The old rationale — "the kernels object is stack-allocated per event, so
inlining the big two would put ~432 KB on the stack" — evaporates at 56 B each.

**`OwnedPointer[List[T]]` is always wrong.** A `List` is already a heap-backed
handle, so the box is a second allocation and a second indirection, and §12
above measures that indirection being *reloaded on every iteration* because the
backend cannot hoist it. 29 of the port's 39 boxes were this shape.

**38 of the port's 39 boxes are gone.** All 15 columns plus
`m_HistStore`/`m_AverageGeometryStore` in `TrackingRecHit2DHeterogeneous`, all 7
in `SiPixelDigisSoA`, all 4 in `SiPixelClustersSoA`, both in
`SiPixelDigiErrorsSoA`, both containers in `CAHitNtupletGeneratorKernels`, `ws_d`
in `gpuVertexFinder`, both in `TimerManager`, `_word`/`_fedId` in
`WordFedAppender`, `_wordFedAppender` in `SiPixelRawToClusterCUDA`,
`_gainForHLTonHost` in `SiPixelGainCalibrationForHLTGPU`, and `m_counters` in
`CAHitNtupletGeneratorOnGPU`.

Build cost of the two behavioural tests fell as a side effect — clustering
704 → 299 MB, vertex finder 363 → 174 MB — and both still pass, the clustering
one still byte-identical to the C++ reference.

Two of these were not merely redundant but actively wrong:

- **`TimerManager`** reached its `Dict`/`List` as `self._storage.unsafe_ptr()[]`
  to mutate through a non-`mut` `self`. That does not work: `ptr[]` is an
  **rvalue**, so every `__setitem__`/`append`/`clear`/`pop` through it was
  rejected — 8 of the file's errors. Unboxing to plain fields and marking the
  mutating methods `mut self` took the file to zero. `top()` now returns a
  `String` copy rather than a borrow, because every caller uses it as a `Dict`
  key while mutating `_storage`, which a live borrow of `_cur` forbids; and
  `finalize` copies the key set out before reaching each value mutably.
- **`SiPixelDigiErrorsSoA.error()`/`c_error()`** returned raw `UnsafePointer`s
  into the box. They have no callers, so they are now `ref` returns.

Only `TypeableOwnedPointer._inner` is left. That one is structural — it *is* the
Event-product mechanism (`HeterogeneousSoA`), not an incidental box — so it needs
a design decision rather than a sweep.

**Sizes below which a box is never worth it.** Measured: a 24 kB struct
(`SiPixelGainForHLTonGPU`) with 2 derefs builds in 316 MB, while 216 kB
(`ZVertexSoA`) exhausted 6 GB. So the compile-time cliff sits somewhere between
24 kB and 216 kB — but that only bounds the *hazard*. The allocation and
indirection are wasted at any size where the type is a handle or the accessor
returns a borrow, which covered every case here.

Where a box *is* still wanted, note what C++'s `unique_ptr` actually buys:
in `TrackingRecHit2DHeterogeneous` it was `Traits::unique_ptr`, i.e. a device
allocation whose address is handed to a view — a GPU concern, not an ownership
one (see above). Read the C++ before assuming a `unique_ptr` means "too big to
inline".

### Origins do not become `noalias`

Nothing vectorises — every variant stays scalar (`vmovss`/`vfmadd231ss`). This
is **not** the FP reduction: a pure elementwise loop over three Spans with
distinct origins and no accumulator is still scalar, and LLVM does not even
emit its usual runtime-overlap-check-plus-vector-loop pair.

So origins are erased before codegen. §4 records that they are compile-time
only and cost nothing at runtime; the corollary is that they also **buy**
nothing — they are a borrow-checking device, not an aliasing hint. Do not
expect a `Span[T, _]` parameter to imply `noalias` to the backend.

### `OwnedPointer` of a large inline struct is a compile-time bomb

`mojo run` on a file exercising the vertex finder consumed **23 GB** and was
OOM-killed, twice, taking the editor with it. The cause is not the code being
compiled: it is `OwnedPointer[T]` where `T` is a struct whose columns are
`InlineArray`, and it scales with the number of times the pointee is used.

Measured with `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`
around `mojo build` (**always cap these**, see the warning below):

| shape | peak RSS |
|---|---|
| 600 kB struct on the stack, `mut` param + a field read | 308 MB |
| `OwnedPointer(WorkSpace())`, **1** deref | 566 MB |
| `OwnedPointer(WorkSpace())`, **2** derefs | **>6 GB** |
| `OwnedPointer(WorkSpace())`, 1 deref + a call | **>6 GB** |
| two `OwnedPointer`s, no call at all | **>6 GB** |

What does **not** matter, all measured:

- **The parameter convention.** `mut T`, `ref T` and `Pointer[mut=True, T, _]`
  all behave identically. A stack struct passed `mut` is 308 MB, so `mut` on a
  large struct is fine — this is not the §12 copy-in/copy-out band.
- **Function calls.** Two derefs and a single `print`, no call, still dies.
- **Binding once.** `ref w = wp[]` then reusing `w` does not help.
- **`Copyable`.** A synthetic 256 kB struct dies with or without it.
- **Nesting the arrays in sub-structs.** Also dies.

**The fix: give the struct heap-backed columns.** `InlineArray[T, N]` →
`List[T]`, allocated `List[T](length=N, fill=…)` in `__init__`. Indexing, `ref`
bindings and capacity are unchanged, and §12 already shows hot loops should
bind Spans regardless — a `List` binds to a Span exactly as an `InlineArray`
does, so there is no cost at the access site.

Applied to `ZVertexSoA` and `WorkSpace`:

| | before | after |
|---|---|---|
| `size_of[ZVertexSoA]()` | 216 kB | **176 B** |
| `size_of[WorkSpace]()` | ~600 kB | **160 B** |
| 2 derefs / 3 derefs | KILLED / KILLED | **303 MB / 321 MB** |
| full vertex-finder pipeline | **23 GB, OOM** | **363 MB** |

No call site changed — `Producer.make` still holds both in `OwnedPointer`s, and
the `ref data = pdata[]` bindings in the algorithms are untouched. C++ declares
these columns inline because the SoA is one `cudaMalloc`'d blob on the device;
that rationale does not transfer to a host serial build.

Then applied to `HistoContainer` (`off` and `bins`), which fixes every user at
once — `TrackingRecHit2DHeterogeneous.m_HistStore`, `TrackSoA`'s two
`HitContainer`s, and the three `CAConstants` containers:

| | before | after |
|---|---|---|
| `size_of[Hist]()` (rechit) | 103,432 B | **56 B** |
| `size_of[TrackSoA.HitContainer]()` | 458,760 B | **56 B** |
| `size_of[CAConstants.TupleMultiplicity]()` | 49,192 B | **56 B** |
| construct a `TrackingRecHit2DHeterogeneous`, 2 × `phiBinner()` | **>6 GB** | **307 MB** |

No call site changed: the external uses are all `hist.off[j]` / `tuples.bins[idx]`
indexing, which reads identically on a `List`. Both behavioural tests still pass
and `GPUClusteringTest` is still byte-identical to the C++ reference.

**Why `m_HistStore` is an `OwnedPointer` at all — it is vestigial.** C++ has
`unique_ptr<Hist> m_HistStore`, but that `unique_ptr` is `Traits::unique_ptr`, a
*policy* type: on the CUDA path it is `make_device_unique` (a `cudaMalloc`), and
line 96 does `m_hist = view->m_hist = m_HistStore.get()` — the point is a stable
**device** address to stash into `TrackingRecHit2DSOAView`. Neither reason
survives here: there is no device, and the self-referential views were removed
(§9, §11, §13). So "it mirrors C++'s `unique_ptr`" is not a reason to keep the
box; with `HistoContainer` now heap-backed the box is merely harmless, and could
be dropped later.

**Status of the other large SoAs**

- `PixelTrack.TrackSoA` — was **measured clean** at 2 derefs (306 MB / 281 MB)
  even at 3.8 MB with `fill=`-initialised nested `InlineArray`s. Why it escaped
  was *never established* — `Copyable`, raw size, and nesting the arrays in
  sub-structs were each tested and each ruled out. Do not assume a large SoA is
  safe by analogy; measure it.
- Remaining `InlineArray`-backed SoAs (e.g. `ScalarSoA`, still
  `InlineArray[Scalar, S]`) are unmeasured. Any of them reached through an
  `OwnedPointer` is a candidate.

**`precompile` cannot see any of this.** It never instantiates `main`, so the
error count stays healthy while the codebase is unbuildable — 306→241 errors
was reported all session with this sitting underneath. Verification needs
`mojo build`, and it must be capped: an uncapped `mojo run`/`mojo build` on an
affected file will take the machine down, not just the compiler. This is the
same failure mode as the `--emit asm` OOM noted above.

---

## 13. DONE — removing `ParamsOnGPU`, the second self-referential view

**Correction to §11's claim that no views remained.** `PixelCPEFast._cpuData`
was one: four `UnsafePointer`s into `PixelCPEFast`'s *own* fields, rebuilt in
both constructors and in `__moveinit__`. `ParamsOnGPU` was its view type. The
count was 4 → 1, not 4 → 0. It is now genuinely 0.

`PixelCPEFast` was also broken independently: `__moveinit__` does not exist in
1.0, so the rebuild-the-pointers-after-move hook that made `_cpuData` sound was
not running at all.

### What changed

`ParamsOnGPU` is deleted. Its five accessors — `commonParams()`, `detParams(i)`,
`layerGeometry()`, `averageGeometry()`, `layer(id)` — moved onto `PixelCPEFast`,
returning refs into its own fields. `_cpuData`, `getCPUProduct()` and the
hand-written move constructor are gone; the move is synthesized.

Consumers take `PixelCPEFast` as a borrow. The chain that changed:

| site | before | after |
|---|---|---|
| `SiPixelRecHitCUDA` | `UnsafePointer(to=es.get[PixelCPEFast]().getCPUProduct())` | `es.get[PixelCPEFast]()` |
| `PixelRecHitGPUKernel.makeHits` | `var cpeParams: UnsafePointer[ParamsOnGPU]` | `cpeParams: PixelCPEFast` |
| `setHitsLayerStart` | three `UnsafePointer[…]` | two `Span`s + `PixelCPEFast` |
| `getHits` | `cpeParams: ParamsOnGPU` | `cpeParams: PixelCPEFast` |
| `TrackingRecHit2DHeterogeneous` | stored `m_cpeParams` by value | field dropped entirely |

Dropping `m_cpeParams` is the notable one: the hit SoA stored it only to serve
two call sites, both in still-un-ported files. Those two now need `cpeParams`
threaded in as a parameter:

```
plugin_PixelTriplets/RiemannFitOnGPU.mojo:73
plugin_PixelTriplets/BrokenLineFitOnGPU.mojo:81
    hhp[].cpeParams().detParams(Int32(hhp[].detectorIndex(hit))).frame.toGlobal(
```

`detParams` now takes `Int` rather than `Int32`, matching every other index
accessor in the port.

### Result

675 → **556** errors. All six files in the RecHits chain are at zero:
`PixelCPEforGPU`, `PixelCPEFast`, `TrackingRecHit2DHeterogeneous`,
`PixelRecHits`, `GPUPixelRecHits`, `SiPixelRecHitCUDA`.

Most of the residue was §6 numeric strictness, arriving exactly as described
there — one conversion fixed uncovers the next (`range` → `__lt__` → `__ne__` →
`__add__` → parameter type), five rounds before it settled.

---

## 14. DONE — `CUDACompat`, and `Pointer` is non-nullable

556 → **449** errors from this one small file: it sits in almost every plugin's
import closure, so its 7 errors were masking ~100 downstream.

### `Pointer` cannot be null

Worth knowing before reaching for it. `Pointer` is non-nullable **by
construction** in 1.0 — the constraint fires at instantiation:

```
constraint failed: Pointer is non-nullable.
To construct a null pointer, use Optional[Pointer] to model nullability.
```

So there is no `Pointer` equivalent of a null `UnsafePointer`. `CUDAStreamType`
is now `Optional[Pointer[NoneType, MutUntrackedOrigin]]` with `cudaStreamDefault
= None`. It is a vestigial placeholder — the serial backend has no streams and
never dereferences it — and it is passed along as a default argument in 5 files
with no unwrapping needed.

### The atomics are not atomics

In the serial backend these six are plain read-modify-write. Five now take
`mut a: Scalar[T1]` instead of a pointer, so the borrow is tracked and 18
`UnsafePointer(to=…)` wrappers disappeared from call sites:

```mojo
CUDACompat.atomicAdd(UnsafePointer(to=noise), Int32(1))   # before
CUDACompat.atomicAdd(noise, Int32(1))                      # after
```

`atomicCAS` is the exception and keeps a pointer, now
`Pointer[mut=True, Scalar[T1], _]` — tracked origin rather than untracked. Its
only two callers (`GPUCACell` lines 105 and 132) CAS on a *reinterpreted pointer
field*, `Pointer(to=self.theOuterNeighbors).bitcast[UInt64]()`, which has no
`Scalar` lvalue to borrow.

### `comptime if` typechecks the branch it does not take

Verified on an isolated probe. Those two `atomicCAS` calls sit inside
`comptime if is_defined["__CUDACC__"]()`, dead in every serial build, and they
*still* constrain the signature. Dead CUDA code is not free — it votes on your
API. (`Pointer.bitcast` also warns: use `unsafe_bitcast`.)

### `comptime if` selects a branch but does **not** narrow the type

The corollary, and the one that decides the framework's design. Dispatching on a
type parameter looks like it should work:

```mojo
def get[T: Movable & Named](ref self) -> ref [self.a] T:
    comptime if T.dtype() == "A":       # folds fine, branch is selected
        return self.a.value()           # error: cannot implicitly convert 'A' to 'T'
```

The comparison folds and the right branch is chosen, but inside it `T` is still
the abstract parameter — the compiler does not learn `T == A`. Every branch
therefore needs `rebind[T](...)`, an unchecked cast. Combined with the rule
above (the untaken branch must typecheck too), *both* branches need one.

There is also no type identity to test against: `T is A` fails with *"'Movable'
does not implement the '__is__' method"*.

**Consequence:** a struct of named per-type fields cannot expose a safe
`get[T]()`. `Variant` is the only construct in 1.0 that both selects by type and
knows the arm's concrete type on the far side, which is why §15 uses it rather
than five named fields — even though named fields would avoid the union
padding.

### Assembly for the real `getHits`: abandoned, and why

`CUDACompat`, `SOARotation`, `File` and `Phase1PixelTopology` were all cleared
to chase this. The build then **ran the machine out of memory** — `mojo build
--emit asm` compiles the whole import closure in one process, was killed with
SIGKILL (exit 137), and took the user's editor down with it.

> **Do not run `mojo build --emit asm` over a driver that imports the full
> MojoSerial graph.** Emit assembly from small standalone probes only.

This chase was a mistake worth recording. The §12 result was already measured on
a probe replicating the hit SoA's 13 columns, types and store pattern exactly;
the real function's assembly would have *confirmed* that, not added to it. Each
file cleared revealed the next (§8), and the marginal value never justified the
cost. If it is ever wanted, the way to get it is to extract `getHits` and its
few dependencies into a standalone file, not to build the real closure.

---

## 18. DONE (raw-decode chain) — `UnsafePointer` lost its defaults

Everything below was verified on isolated probes against this toolchain, not
recalled. This is the single largest remaining class, and it is the one the
project wants gone anyway.

### The two breaking changes

`UnsafePointer[T]` no longer supplies a default `origin`, so a bare spelling is
an incomplete type:

```
error: failed to infer parameter 'origin', specify the parameter or use '_' or '...'
```

and its `mut=` keyword parameter is gone, because mutability now rides on the
origin:

```
error: unexpected keyword parameter 'mut'
```

19 sites hit the first, 6 the second. They are not independent bugs; they are
one API change demanding that every pointer name where it points.

### `Pointer` is now a full replacement

`List.unsafe_ptr()` already returns **`Pointer[T, origin_of(l)]`** — not
`UnsafePointer`. Forcing a type mismatch prints it:

```
error: cannot implicitly convert 'Pointer[Int, origin_of(l)]' value to 'Int'
```

`Pointer` supports offset indexing, arithmetic and bitcast, all three verified
running:

```mojo
var p = l.unsafe_ptr()
print(p[0])                  # ok
var q = p + 1                # ok
var b = p.unsafe_bitcast[UInt8]()   # ok
```

So there is no capability reason left to keep `UnsafePointer` anywhere.

### But the `unsafe_` names are the API telling you which tool to pick

Two of those three warn:

```
warning: positional `__getitem__` is deprecated, use `unsafe_offset=` instead
warning: 'bitcast' is deprecated, use 'unsafe_bitcast' instead
```

The stdlib is steering `Pointer` toward *one object*, and marking array-shaped
use as the unsafe path. Read that as the selection rule, not as noise:

| what the pointer means | port it to | deref |
|---|---|---|
| a view of **one** object | `Pointer[T, origin]` | `p[]` |
| a view of **many** (a buffer, a column) | `Span[T, origin]` | `s[i]` |
| an **owned** buffer | `List[T]` | `l[i]` |

Applied to the remaining sites this is exhaustive — nothing needs an `unsafe_`
spelling, because every array-shaped pointer here is a `Span` and every
single-object pointer is a `Pointer`.

### Where the origin comes from decides how hard a site is

The mechanical ones borrow from something the caller already owns, so the origin
is inferred and the call site does not change. The hard ones are **struct
fields**, where the field must name an origin that outlives it — and if the
storage is a *sibling field* of the same struct, that is the self-referential
view problem again (§11, §13) and the answer is to restructure, not to
parameterize. Remember `Pointer` is non-nullable (§14): a field modelling "not
pointing yet" needs `Optional[Pointer[…]]` or an index sentinel, not a null.

### The raw-decode chain, done — 45 → 29

`FEDRawData` → `FEDHeader`/`FEDTrailer` → `ErrorChecker` → `WordFedAppender` →
`fillHitsModuleStart`. One origin, owned by the `FEDRawDataCollection` in the
Event, flowing the whole way. All six files are at zero and **14
`UnsafePointer`s are gone** — none of them merely re-spelled:

- **`FEDHeader`/`FEDTrailer` hold the struct by value.** C++ keeps
  `const fedh_struct*` into the buffer; both structs are 8 bytes and
  `TrivialRegisterPassable`, every accessor is a pure read, and `ErrorChecker`
  constructs → reads → drops inside one call with no write to the buffer in
  between. So the pointer is *deleted*, not re-spelled, and `Defaultable`
  survives. The reinterpret moves out to `ErrorChecker`
  (`header.unsafe_bitcast[FedhType]()[]`), which is the one genuinely unsafe
  act in the chain and now carries an `unsafe_` name. Divergence: C++ re-reads
  the buffer per accessor, this snapshots at construction.
- **`fillHitsModuleStart` takes `Span`s**, which is what its callers already
  returned and what `blockPrefixScan` already took — the pointer declaration
  was the only mismatch left.
- **The self-aliased scan.** C++ calls
  `blockPrefixScan(moduleStart + 1, moduleStart + 1, 1024)` — one buffer as
  both `ci` and `co`. With `Span`s that is one buffer borrowed twice, once
  mutably. The existing 2-arg in-place overload computes exactly that, so
  `blockPrefixScan(moduleStart[1:], 1024)` is both legal and more honest.

### `Pointer` catches what `UnsafePointer` laundered

`WordFedAppender.initializeWordFed` was declared `self` — immutable — and wrote
into `self._word`/`self._fedId` through `unsafe_ptr()`. `UnsafePointer` erased
that; `Pointer` refused it, and C++ has the method non-const. **A real bug,
surfaced by the conversion rather than by any test.** Expect more of these: the
pointer errors were masking the method *bodies*, so this chain went 45 → 54
before it went to 29, and every one of the newly-revealed errors was a genuine
pre-existing defect (unchecked `int`→`uint8_t` narrowing in seven accessors,
`range()` over a `UInt32` constant, four non-transferred `append`s).

### Three more 1.0 API facts, measured

- **`memcpy` is keyword-only**: `memcpy[T](*, dest:, src:, count:)`, taking
  `Optional[Pointer[T, origin]]`. Positional calls fail with the confusing
  "missing required argument: 'dest'".
- **`Span` has no ptr+length constructor** — only `__init__()` and
  `__init__(other: Span)`. So a pointer-plus-count pair that arrives from
  outside *cannot* be turned into a `Span`; it stays a `Pointer`. This is what
  keeps `initializeWordFed(src)` and `ErrorChecker`'s three parameters pointers.
- **Parametric `comptime` aliases work**:
  `comptime Iterator[origin: Origin] = _ListIter[UChar, origin]`. Needed because
  `_ListIter` is now `[mut: Bool, //, T: Copyable, origin: Origin[mut=mut]]`,
  and `Self.Data.T` fails its `Copyable` bound — `List`'s `T` carries the
  *declared* `Movable` bound, not the concrete element's traits. Name the
  element type concretely.

### `precompile` is blind to more than codegen

`mojo build` on `bin/main.mojo` reports errors `precompile` does not — e.g.
`SimpleVector … doesn't conform to 'Movable'` and `field 'device_theCells_' has
non-'Deinitable' type 'List[GPUCACell]'`. Use the real build as the final gate,
not the error count.