#include <Eigen/Core>
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"
#include <memory>
#include <cstdlib>
#include <Eigen/Dense>

// Test SoA stores and view.
// Use cases
// Multiple stores in a buffer
// Scalars, Columns of scalars and of Eigen vectors
// View to each of them, from one and multiple stores.

GENERATE_SOA_LAYOUT_AND_VIEW(SoA1LayoutTemplate,
                             SoA1ViewTemplate,
                             // predefined static scalars
                             // size_t size;
                             // size_t alignment;

                             // columns: one value per element
                             SOA_COLUMN(double, x),
                             SOA_COLUMN(double, y),
                             SOA_COLUMN(double, z),
                             SOA_COLUMN(double, sum),
                             SOA_COLUMN(double, prod),
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
                             SOA_EIGEN_COLUMN(Eigen::Vector3d, r),
                             SOA_COLUMN(uint16_t, color),
                             SOA_COLUMN(int32_t, value),
                             SOA_COLUMN(double *, py),
                             SOA_COLUMN(uint32_t, count),
                             SOA_COLUMN(uint32_t, anotherCount),

                             // scalars: one value for the whole structure
                             SOA_SCALAR(const char *, description),
                             SOA_SCALAR(uint32_t, someNumber))

using SoA1Layout = SoA1LayoutTemplate<>;
using SoA1View = SoA1ViewTemplate<>;

// A partial view (artificial mix of store and view)
GENERATE_SOA_VIEW(SoA1View2GTemplate,
                  SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(SoA1Layout, soa1), SOA_VIEW_LAYOUT(SoA1View, soa1v)),
                  SOA_VIEW_VALUE_LIST(SOA_VIEW_VALUE(soa1, x),
                                      SOA_VIEW_VALUE(soa1v, y),
                                      SOA_VIEW_VALUE(soa1, color),
                                      SOA_VIEW_VALUE(soa1v, value),
                                      SOA_VIEW_VALUE(soa1v, count),
                                      SOA_VIEW_VALUE(soa1, anotherCount),
                                      SOA_VIEW_VALUE(soa1v, description),
                                      SOA_VIEW_VALUE(soa1, someNumber)))

using SoA1View2G = SoA1View2GTemplate<>;

// Same partial view, yet const.
GENERATE_SOA_CONST_VIEW(SoA1View2Gconst,
                        SOA_VIEW_LAYOUT_LIST(SOA_VIEW_LAYOUT(SoA1Layout, soa1), SOA_VIEW_LAYOUT(SoA1View, soa1v)),
                        SOA_VIEW_VALUE_LIST(SOA_VIEW_VALUE(soa1, x),
                                            SOA_VIEW_VALUE(soa1v, y),
                                            SOA_VIEW_VALUE(soa1, a),
                                            SOA_VIEW_VALUE(soa1, b),
                                            SOA_VIEW_VALUE(soa1, r),
                                            SOA_VIEW_VALUE(soa1, color),
                                            SOA_VIEW_VALUE(soa1v, value),
                                            SOA_VIEW_VALUE(soa1v, count),
                                            SOA_VIEW_VALUE(soa1, anotherCount),
                                            SOA_VIEW_VALUE(soa1v, description),
                                            SOA_VIEW_VALUE(soa1, someNumber)))

// Parameter reusing kernels.  The disassembly will indicate whether the compiler uses the wanted cache hits and uses
// `restrict` hints avoid multiple reduce loads.
// The PTX can be obtained using -ptx insterad of -c when compiling.
template <typename T>
__device__ void addAndMulTemplate(T soa, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto si = soa[idx];
  si.sum() = si.x() + si.y();
  si.prod() = si.x() * si.y();
}

__global__ void aAMDef(SoA1ViewTemplate<cms::soa::CacheLineSize::defaultSize,
                                        cms::soa::AlignmentEnforcement::Relaxed,
                                        cms::soa::RestrictQualify::Disabled> soa,
                       size_t size) {
  addAndMulTemplate(soa, size);
}

__global__ void aAMRestrict(SoA1ViewTemplate<cms::soa::CacheLineSize::defaultSize,
                                             cms::soa::AlignmentEnforcement::Relaxed,
                                             cms::soa::RestrictQualify::Enabled> soa,
                            size_t size) {
  addAndMulTemplate(soa, size);
}

const size_t size = 10000;

int main() {
  // Allocate buffer
  std::unique_ptr<std::byte, decltype(&std::free)> buffer(
      static_cast<std::byte *>(std::aligned_alloc(SoA1Layout::defaultAlignment, SoA1Layout::computeDataSize(size))),
      std::free);
  SoA1Layout soa1(buffer.get(), size);
  SoA1View soa1view(soa1);
  SoA1View2G soa1v2g(soa1, soa1view);
  SoA1View2Gconst soa1v2gconst(soa1, soa1view);
  // Write to view
  for (size_t i = 0; i < size; i++) {
    auto s = soa1view[i];
    s.x = 1.0 * i;
    s.y = 2.0 * i;
    s.z = 3.0 * i;
    s.color() = i;
    s.a()(0) = 1.0 * i;
    s.a()(1) = 2.0 * i;
    s.a()(2) = 3.0 * i;
    s.b()(0) = 3.0 * i;
    s.b()(1) = 2.0 * i;
    s.b()(2) = 1.0 * i;
    s.r() = s.a().cross(s.b());
  }
  // Check direct read back
  for (size_t i = 0; i < size; i++) {
    auto s = soa1view[i];
    assert(s.x() == 1.0 * i);
    assert(s.y() == 2.0 * i);
    assert(s.z() == 3.0 * i);
    assert(s.color() == i);
    assert(s.a()(0) == 1.0 * i);
    assert(s.a()(1) == 2.0 * i);
    assert(s.a()(2) == 3.0 * i);
    assert(s.b()(0) == 3.0 * i);
    assert(s.b()(1) == 2.0 * i);
    assert(s.b()(2) == 1.0 * i);
    assert(s.r() == s.a().cross(s.b()));
  }
  // Check readback through other views
  for (size_t i = 0; i < size; i++) {
    auto sv = soa1view[i];
    auto sv2g = soa1v2g[i];
    auto sv2gc = soa1v2gconst[i];
    assert(sv.x() == 1.0 * i);
    assert(sv.y() == 2.0 * i);
    assert(sv.z() == 3.0 * i);
    assert(sv.color() == i);
    assert(sv2g.x() == 1.0 * i);
    assert(sv2g.y() == 2.0 * i);
    assert(sv2g.color() == i);
    assert(sv2gc.x() == 1.0 * i);
    assert(sv2gc.y() == 2.0 * i);
    assert(sv2gc.color() == i);
  }

  // Validation of range checking
  try {
    // Get a view like the default, except for range checking
    SoA1ViewTemplate<SoA1View::byteAlignment,
                     SoA1View::alignmentEnforcement,
                     SoA1View::restrictQualify,
                     cms::soa::RangeChecking::Enabled>
        soa1viewRangeChecking(soa1);
    // This should throw an exception
    [[maybe_unused]] auto si = soa1viewRangeChecking[soa1viewRangeChecking.soaMetadata().size()];
    assert(false);
  } catch (const std::out_of_range &) {
  }
}