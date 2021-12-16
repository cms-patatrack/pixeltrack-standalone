#ifndef FWCore_Utilities_ProductEDPutToken_h
#define FWCore_Utilities_ProductEDPutToken_h

#include "Framework/EDPutToken.h"
#include "Framework/Host.h"

namespace edm {
  template <typename T>
  class Product;

  template <typename T>
  struct ProductEDPutToken {
    static auto toToken(edm::EDPutTokenT<Product<T>> const& token) {
      return EDPutTokenT<T>(token.index());
    }

    static auto toProductToken(edm::EDPutTokenT<T> const& token) {
      return EDPutTokenT<Product<T>>(token.index());
    }

    static auto toToken(edm::EDPutTokenT<Host<T>> const& token) {
      return EDPutTokenT<T>(token.index());
    }

    static auto toHostToken(edm::EDPutTokenT<T> const& token) {
      return EDPutTokenT<Host<T>>(token.index());
    }
  };
}

#endif
