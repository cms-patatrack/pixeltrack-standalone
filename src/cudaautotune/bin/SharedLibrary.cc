// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     SharedLibrary
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Apr  5 15:30:15 EDT 2007
//

// system include files
#include <string> /*needed by the following include*/
#include <dlfcn.h>
#include <cerrno>
#include <stdexcept>

// user include files
#include "SharedLibrary.h"

namespace edmplugin {
  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  SharedLibrary::SharedLibrary(const std::string& iName)
      : libraryHandle_(::dlopen(iName.c_str(), RTLD_LAZY | RTLD_GLOBAL)), path_(iName) {
    if (libraryHandle_ == nullptr) {
      char const* err = dlerror();
      if (err == nullptr) {
        throw std::runtime_error("unable to load " + iName);
      }
      throw std::runtime_error("unable to load " + iName + " because " + err);
    }
  }

  // SharedLibrary::SharedLibrary(const SharedLibrary& rhs)
  // {
  //    // do actual copying here;
  // }

  SharedLibrary::~SharedLibrary() {}

  //
  // assignment operators
  //
  // const SharedLibrary& SharedLibrary::operator=(const SharedLibrary& rhs)
  // {
  //   //An exception safe implementation is
  //   SharedLibrary temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //

  //
  // const member functions
  //
  bool SharedLibrary::symbol(const std::string& iSymbolName, void*& iSymbol) const {
    if (libraryHandle_ == nullptr) {
      return false;
    }
    iSymbol = dlsym(libraryHandle_, iSymbolName.c_str());
    return (iSymbol != nullptr);
  }

  //
  // static member functions
  //
}  // namespace edmplugin
