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

GENERATE_SOA_LAYOUT(SoA1LayoutTemplate,
  // predefined static scalars
  // size_t size;
  // size_t alignment;

  // columns: one value per element
  SOA_COLUMN(double, x),
  SOA_COLUMN(double, y),
  SOA_COLUMN(double, z),
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
  SOA_SCALAR(uint32_t, someNumber)
);

using SoA1Layout = SoA1LayoutTemplate<>;

// A 1 to 1 view of the store (except for unsupported types).
generate_SoA_view(SoA1ViewTemplate,
  SOA_VIEW_LAYOUT_LIST(
    SOA_VIEW_LAYOUT(SoA1Layout, soa1)
  ),
  SOA_VIEW_VALUE_LIST(
    SOA_VIEW_VALUE(soa1, x),
    SOA_VIEW_VALUE(soa1, y),
    SOA_VIEW_VALUE(soa1, z),
    SOA_VIEW_VALUE(soa1, color),
    SOA_VIEW_VALUE(soa1, value),
    SOA_VIEW_VALUE(soa1, py),
    SOA_VIEW_VALUE(soa1, count),
    SOA_VIEW_VALUE(soa1, anotherCount), 
    SOA_VIEW_VALUE(soa1, description),
    SOA_VIEW_VALUE(soa1, someNumber)
  )
);

using SoA1View = SoA1ViewTemplate<>;

// A partial view (artificial mix of store and view)
generate_SoA_view(SoA1View2GTemplate,
  SOA_VIEW_LAYOUT_LIST(
    SOA_VIEW_LAYOUT(SoA1Layout, soa1),
    SOA_VIEW_LAYOUT(SoA1View, soa1v)
  ),
  SOA_VIEW_VALUE_LIST(
    SOA_VIEW_VALUE(soa1, x),
    SOA_VIEW_VALUE(soa1v, y),
    SOA_VIEW_VALUE(soa1, color),
    SOA_VIEW_VALUE(soa1v, value),
    SOA_VIEW_VALUE(soa1v, count),
    SOA_VIEW_VALUE(soa1, anotherCount), 
    SOA_VIEW_VALUE(soa1v, description),
    SOA_VIEW_VALUE(soa1, someNumber)
  )
);

using SoA1View2G = SoA1View2GTemplate<>;



// Same partial view, yet const.
generate_SoA_const_view(SoA1View2Gconst,
  SOA_VIEW_LAYOUT_LIST(
    SOA_VIEW_LAYOUT(SoA1Layout, soa1),
    SOA_VIEW_LAYOUT(SoA1View, soa1v)
  ),
  SOA_VIEW_VALUE_LIST(
    SOA_VIEW_VALUE(soa1, x),
    SOA_VIEW_VALUE(soa1v, y),
/* Eigen columns are not supported in views.    
    SoA_view_value(soa1, a, a),
    SoA_view_value(soa1, b, b),
    SoA_view_value(soa1, r, r), */
    SOA_VIEW_VALUE(soa1, color),
    SOA_VIEW_VALUE(soa1v, value),
    SOA_VIEW_VALUE(soa1v, count),
    SOA_VIEW_VALUE(soa1, anotherCount), 
    SOA_VIEW_VALUE(soa1v, description),
    SOA_VIEW_VALUE(soa1, someNumber)
  )
);

const size_t size=10000;

int main() {
  // Allocate buffer
  std::unique_ptr<std::byte, decltype(&std::free)> buffer(
    static_cast<std::byte*>(std::aligned_alloc(SoA1Layout::defaultAlignment, SoA1Layout::computeDataSize(size))),
    std::free);
  SoA1Layout soa1(buffer.get(), size);
  SoA1View soa1view (soa1);
  SoA1View2G soa1v2g (soa1, soa1view);
  SoA1View2Gconst soa1v2gconst (soa1, soa1view);
  // Write to view
  for (size_t i=0; i < size; i++) {
    auto s = soa1view[i];
    s.x = 1.0 * i;
    s.y = 2.0 * i;
    s.z = 3.0 * i;
    s.color() = i;
    // TODO: re-enable when support of eigen is added to views.
    /*s.a()(0) = 1.0 * i;
    s.a()(1) = 2.0 * i;
    s.a()(2) = 3.0 * i;
    s.b()(0) = 3.0 * i;
    s.b()(1) = 2.0 * i;
    s.b()(2) = 1.0 * i;
    s.r() = s.a().cross(s.b());*/
  }
  // Check direct read back
  for (size_t i=0; i < size; i++) {
    auto s = soa1view[i];
    assert(s.x() == 1.0 * i);
    assert(s.y() == 2.0 * i);
    assert(s.z() == 3.0 * i);
    assert(s.color() == i);
    // TODO: re-enable when support of eigen is added to views.
    /*assert(s.a()(0) == 1.0 * i);
    assert(s.a()(1) == 2.0 * i);
    assert(s.a()(2) == 3.0 * i);
    assert(s.b()(0) == 3.0 * i);
    assert(s.b()(1) == 2.0 * i);
    assert(s.b()(2) == 1.0 * i);
    assert(s.r() == s.a().cross(s.b()));*/
  }
  // Check readback through other views
  for (size_t i=0; i < size; i++) {
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
}