#ifndef FWCore_Utilities_EDPutToken_h
#define FWCore_Utilities_EDPutToken_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     EDPutToken
//
/**\class EDPutToken EDPutToken.h "FWCore/Utilities/interface/EDPutToken.h"

 Description: A Token used to put data into the EDM

 Usage:
    A EDPutToken is created by calls to 'produces'from an EDProducer or EDFilter.
 The EDPutToken can then be used to quickly put data into the edm::Event, edm::LuminosityBlock or edm::Run.
 
The templated form, EDPutTokenT<T>, is the same as EDPutToken except when used to get data the framework
 will skip checking that the type being requested matches the type specified during the 'produces'' call.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 18 Sep 2017 17:54:11 GMT
//

// system include files
#include <string>
#include <utility>

// user include files

// forward declarations
namespace edm {
  template <typename T>
  class EDPutTokenT;
  class ProductRegistry;

  /// The type-erased token.  See EDGetToken for why it carries a type name.
  class EDPutToken {
    friend class ProductRegistry;

  public:
    using value_type = unsigned int;

    EDPutToken() : m_value{s_uninitializedValue} {}

    template <typename T>
    EDPutToken(EDPutTokenT<T> iOther) : m_value{iOther.m_value} {}

    // ---------- const member functions ---------------------
    value_type index() const { return m_value; }
    bool isUninitialized() const { return m_value == s_uninitializedValue; }
    std::string const& typeName() const { return m_typeName; }

  private:
    //for testing
    friend class TestEDPutToken;

    static const unsigned int s_uninitializedValue = 0xFFFFFFFF;

    explicit EDPutToken(unsigned int iValue) : m_value(iValue) {}
    EDPutToken(unsigned int iValue, std::string iTypeName) : m_value(iValue), m_typeName(std::move(iTypeName)) {}

    // ---------- member data --------------------------------
    value_type m_value;
    std::string m_typeName;
  };

  template <typename T>
  class EDPutTokenT {
    friend class ProductRegistry;
    friend class EDPutToken;

  public:
    using value_type = EDPutToken::value_type;

    EDPutTokenT() : m_value{s_uninitializedValue} {}

    // ---------- const member functions ---------------------
    value_type index() const { return m_value; }
    bool isUninitialized() const { return m_value == s_uninitializedValue; }

  private:
    //for testing
    friend class TestEDPutToken;

    static const unsigned int s_uninitializedValue = 0xFFFFFFFF;

    explicit EDPutTokenT(unsigned int iValue) : m_value(iValue) {}

    // ---------- member data --------------------------------
    value_type m_value;
  };
}  // namespace edm

#endif
