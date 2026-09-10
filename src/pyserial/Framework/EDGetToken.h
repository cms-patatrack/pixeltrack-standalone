#ifndef FWCore_Utilities_EDGetToken_h
#define FWCore_Utilities_EDGetToken_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     EDGetToken
//
/**\class EDGetToken EDGetToken.h "FWCore/Utilities/interface/EDGetToken.h"

 Description: A Token used to get data from the EDM

 Usage:
    A EDGetToken is created by calls to 'consumes' or 'mayConsume' from an EDM module.
 The EDGetToken can then be used to quickly retrieve data from the edm::Event, edm::LuminosityBlock or edm::Run.
 
The templated form, EDGetTokenT<T>, is the same as EDGetToken except when used to get data the framework
 will skip checking that the type being requested matches the type specified during the 'consumes' or 'mayConsume' call.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 03 Apr 2013 17:54:11 GMT
//

// system include files
#include <string>
#include <utility>

// user include files

// forward declarations
namespace edm {
  template <typename T>
  class EDGetTokenT;
  class ProductRegistry;

  /// The type-erased token.  It is what a Python module holds: there the type
  /// cannot travel as a template argument, so it travels as a name, which the
  /// generated dispatch matches to pick the product's bindings.  A token made
  /// from an EDGetTokenT<T> carries no name, because C++ never needs one -- the
  /// type is already known at the call site.
  class EDGetToken {
    friend class ProductRegistry;

  public:
    EDGetToken() : m_value{s_uninitializedValue} {}

    template <typename T>
    EDGetToken(EDGetTokenT<T> iOther) : m_value{iOther.m_value} {}

    // ---------- const member functions ---------------------
    unsigned int index() const { return m_value; }
    bool isUninitialized() const { return m_value == s_uninitializedValue; }
    std::string const& typeName() const { return m_typeName; }

  private:
    //for testing
    friend class TestEDGetToken;

    static const unsigned int s_uninitializedValue = 0xFFFFFFFF;

    explicit EDGetToken(unsigned int iValue) : m_value(iValue) {}
    EDGetToken(unsigned int iValue, std::string iTypeName) : m_value(iValue), m_typeName(std::move(iTypeName)) {}

    // ---------- member data --------------------------------
    unsigned int m_value;
    std::string m_typeName;
  };

  template <typename T>
  class EDGetTokenT {
    friend class ProductRegistry;
    friend class EDGetToken;

  public:
    EDGetTokenT() : m_value{s_uninitializedValue} {}

    // ---------- const member functions ---------------------
    unsigned int index() const { return m_value; }
    bool isUninitialized() const { return m_value == s_uninitializedValue; }

  private:
    //for testing
    friend class TestEDGetToken;

    static const unsigned int s_uninitializedValue = 0xFFFFFFFF;

    explicit EDGetTokenT(unsigned int iValue) : m_value(iValue) {}

    // ---------- member data --------------------------------
    unsigned int m_value;
  };
}  // namespace edm

#endif
