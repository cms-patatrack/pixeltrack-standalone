#ifndef FWCore_Utilities_Host_h
#define FWCore_Utilities_Host_h

namespace edm {
  /**
   * Type helper to let the portability system to know that a product
   * is read from "host memory"
   */
  template <typename T>
  struct Host {
    using type = T;
  };
}

#endif
