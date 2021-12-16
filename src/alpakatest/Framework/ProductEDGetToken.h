#ifndef FWCore_Utilities_ProductEDGetToken_h
#define FWCore_Utilities_ProductEDGetToken_h

#include "Framework/EDGetToken.h"
#include "Framework/Host.h"

namespace edm {
  template <typename T>
  class Product;

  template <typename T>
  struct ProductEDGetToken {
    static auto toToken(edm::EDGetTokenT<Product<T>> const& token) {
      return EDGetTokenT<T>(token.index());
    }

    static auto toProductToken(edm::EDGetTokenT<T> const& token) {
      return EDGetTokenT<Product<T>>(token.index());
    }

    static auto toToken(edm::EDGetTokenT<Host<T>> const& token) {
      return EDGetTokenT<T>(token.index());
    }

    static auto toHostToken(edm::EDGetTokenT<T> const& token) {
      return EDGetTokenT<Host<T>>(token.index());
    }
  };
}

#endif
