// Generate the nanobind bindings for every type stored in the Event, from the
// ROOT dictionaries.
//
//   root -l -b -q 'tools/generate_bindings.C("libdict.so", "out.cc",
//                                            "ZVertexHeterogeneous,PixelTrackHeterogeneous")'
//
// The arguments are the *product* types.  Everything reachable from them --
// members, members of members -- is reflected too, so a product is bound whole
// whether or not any Python module happens to use it today.  Nothing about the
// bindings is written by hand: rename a member or add a method and the
// bindings follow on the next build.
//
// What is emitted, per class:
//
//   scalar member          -> def_rw
//   T member[N]            -> def_prop_ro giving a zero-copy numpy view
//   class member           -> def_prop_ro giving a reference, class bound too
//   static const scalar    -> def_prop_ro_static
//   method                 -> def, through a lambda
//
// Methods are emitted as lambdas rather than member-function pointers on
// purpose: several return `auto`, which cannot be spelled in the cast a member
// pointer needs, but a lambda lets the compiler deduce it.  It also sidesteps
// const/non-const overload pairs, where taking the address would be ambiguous.
//
// Anything that cannot be bound is skipped, recorded as a comment in the
// generated file and warned about on stderr.  A binding that silently
// disappears is worse than an error, and one that silently reports the wrong
// thing -- an array bound as though it were its first element, say -- is worse
// still.

#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

  std::vector<std::string> split(const std::string& text, char separator = ',') {
    std::vector<std::string> items;
    std::string item;
    for (const char c : text) {
      if (c == separator) {
        if (!item.empty())
          items.push_back(item);
        item.clear();
      } else if (!std::isspace(static_cast<unsigned char>(c))) {
        item += c;
      }
    }
    if (!item.empty())
      items.push_back(item);
    return items;
  }

  /// Types nanobind converts by value without any help, and that numpy has a
  /// dtype for.
  const std::set<std::string>& scalars() {
    static const std::set<std::string> known = {"bool",
                                                "char",
                                                "signed char",
                                                "unsigned char",
                                                "short",
                                                "unsigned short",
                                                "int",
                                                "unsigned int",
                                                "long",
                                                "unsigned long",
                                                "long long",
                                                "unsigned long long",
                                                "size_t",
                                                "int8_t",
                                                "uint8_t",
                                                "int16_t",
                                                "uint16_t",
                                                "int32_t",
                                                "uint32_t",
                                                "int64_t",
                                                "uint64_t",
                                                "float",
                                                "double"};
    return known;
  }

  bool isScalar(const std::string& type) { return scalars().count(type) != 0; }

  /// ROOT normalises the standard library's namespace away: a member of type
  /// std::vector<unsigned int> is reported as "vector<unsigned int>", and that
  /// name is what gets emitted as C++.  Put std:: back, or the generated file
  /// does not compile -- and two spellings of one class would otherwise be
  /// collected, and bound, twice.
  std::string qualify(const std::string& type) {
    static const std::set<std::string> stdTemplates = {"vector",
                                                       "array",
                                                       "pair",
                                                       "tuple",
                                                       "map",
                                                       "set",
                                                       "span",
                                                       "unique_ptr",
                                                       "shared_ptr",
                                                       "basic_string"};
    const auto templateArgs = type.find('<');
    if (templateArgs == std::string::npos)
      return type == "string" ? "std::string" : type;
    if (stdTemplates.count(type.substr(0, templateArgs)))
      return "std::" + type;
    return type;
  }

  std::string strip(std::string type) {
    // trailing/leading qualifiers ROOT prints that do not change the binding
    const std::string prefix = "const ";
    while (type.rfind(prefix, 0) == 0)
      type = type.substr(prefix.size());
    // ROOT keeps __restrict in a return type, as in "CommonParams&__restrict",
    // where it says something about the pointer and nothing about the class.
    for (const std::string qualifier : {"__restrict__", "__restrict"}) {
      const auto at = type.find(qualifier);
      if (at != std::string::npos) {
        type.erase(at, qualifier.size());
        break;
      }
    }
    while (!type.empty() && (type.back() == ' ' || type.back() == '&'))
      type.pop_back();
    return qualify(type);
  }

  /// A C++ identifier made from a type name, for use as a local variable.
  std::string sanitise(const std::string& type) {
    std::string out;
    for (const char c : type)
      out += (std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
    return out;
  }

  /// "eigenSoA::ScalarSoA<float,32768>" -> "ScalarSoA<float,32768>"
  ///
  /// TClass names a class with its namespaces; TMethod names a constructor
  /// without them, so recognising a constructor means comparing against this.
  std::string unqualified(const std::string& type) {
    const auto templateArgs = type.find('<');
    const auto cut = type.rfind("::", templateArgs == std::string::npos ? std::string::npos : templateArgs);
    return cut == std::string::npos ? type : type.substr(cut + 2);
  }

  /// "eigenSoA::ScalarSoA<float,32768>" -> "ScalarSoA_float_32768"
  /// The Python-visible name: unique, and legible enough to use from a script.
  std::string pythonName(const std::string& type) {
    std::string bare = type;
    const auto ns = bare.find('<');
    const auto cut = bare.rfind("::", ns == std::string::npos ? std::string::npos : ns);
    if (cut != std::string::npos)
      bare = bare.substr(cut + 2);
    std::string out;
    for (const char c : bare) {
      if (std::isalnum(static_cast<unsigned char>(c)))
        out += c;
      else if (!out.empty() && out.back() != '_')
        out += '_';
    }
    while (!out.empty() && out.back() == '_')
      out.pop_back();
    return out;
  }

  /// The X of a constructor taking std::unique_ptr<X>&&, or "".
  ///
  /// This is how a wrapper says what it owns.  HeterogeneousSoA keeps its
  /// unique_ptr private and hands the payload out through a get() whose return
  /// type is `auto`, so the constructor is the only place the dictionary spells
  /// the payload type out.
  std::string ownedPayload(TClass* cls) {
    for (auto* entry : *cls->GetListOfMethods()) {
      auto* method = static_cast<TMethod*>(entry);
      if (std::string(method->GetName()) != unqualified(cls->GetName()) &&
          std::string(method->GetName()) != cls->GetName())
        continue;
      if (!method->GetListOfMethodArgs() || method->GetListOfMethodArgs()->GetSize() != 1)
        continue;
      const std::string argType =
          static_cast<TMethodArg*>(method->GetListOfMethodArgs()->At(0))->GetFullTypeName();
      const std::string tag = "unique_ptr<";
      const auto at = argType.find(tag);
      if (at == std::string::npos)
        continue;
      std::string payload = argType.substr(at + tag.size());
      const auto end = payload.rfind('>');
      if (end != std::string::npos)
        payload = payload.substr(0, end);
      const auto comma = payload.find(',');
      if (comma != std::string::npos)
        payload = payload.substr(0, comma);
      while (!payload.empty() && payload.back() == ' ')
        payload.pop_back();
      return payload;
    }
    return "";
  }

  /// "X*" -> "X"; anything else unchanged.  What a method hands back by
  /// pointer is the same class as what it hands back by reference.
  std::string pointee(std::string type) {
    while (!type.empty() && (type.back() == '*' || type.back() == ' '))
      type.pop_back();
    return type;
  }

  /// The element type of a span of scalars, or "".
  ///
  /// A span is a pointer and a length together, which is exactly what a
  /// zero-copy view needs and exactly what a bare pointer cannot say.  A
  /// format that spells its columns this way needs no other help from the
  /// generator, and none of this knows which format that is.
  ///
  /// std::span is what a format should use; edm::Span is accepted too, since
  /// CMSSW has one in FWCore/Utilities.  The rule is about the shape rather
  /// than the spelling.
  std::string spanElement(const std::string& type) {
    for (const std::string prefix : {"span<", "std::span<", "Span<", "edm::Span<"}) {
      if (type.rfind(prefix, 0) != 0)
        continue;
      std::string element = type.substr(prefix.size());
      const auto end = element.rfind('>');
      if (end != std::string::npos)
        element = element.substr(0, end);
      // span<T,N> carries its extent as a second argument; the dynamic extent
      // is what a column has.
      const auto comma = element.find(',');
      if (comma != std::string::npos)
        element = element.substr(0, comma);
      element = strip(element);
      const std::string constPrefix = "const ";
      if (element.rfind(constPrefix, 0) == 0)
        element = element.substr(constPrefix.size());
      while (!element.empty() && element.back() == ' ')
        element.pop_back();
      return isScalar(element) ? element : "";
    }
    return "";
  }

  /// The element type of a std::vector of scalars, or "".
  ///
  /// A vector is storage plus a length, which is what a numpy view needs, but
  /// it has neither a public array member nor a fixed extent, so the array rule
  /// cannot see it.  Recognising the one standard container that products are
  /// commonly made of is a rule about the language, not about any product.
  std::string vectorElement(const std::string& type) {
    for (const std::string prefix : {"std::vector<", "vector<"}) {
      if (type.rfind(prefix, 0) != 0)
        continue;
      std::string element = type.substr(prefix.size());
      const auto end = element.rfind('>');
      if (end != std::string::npos)
        element = element.substr(0, end);
      const auto comma = element.find(',');
      if (comma != std::string::npos)
        element = element.substr(0, comma);
      while (!element.empty() && element.back() == ' ')
        element.pop_back();
      return isScalar(element) ? element : "";
    }
    return "";
  }

  bool hasDictionary(const std::string& type) {
    TClass* cls = TClass::GetClass(type.c_str());
    return cls != nullptr && cls->GetListOfDataMembers() != nullptr;
  }

  /// The classes reachable from `type` through its public data members,
  /// deepest first, so a class is registered before anything that refers to it.
  void collect(const std::string& type, std::vector<std::string>& order, std::set<std::string>& seen) {
    if (!seen.insert(type).second)
      return;
    TClass* cls = TClass::GetClass(type.c_str());
    if (!cls)
      return;
    if (!vectorElement(type).empty()) {
      order.push_back(type);
      return;
    }
    if (const std::string payload = ownedPayload(cls); !payload.empty() && hasDictionary(payload))
      collect(payload, order, seen);
    for (auto* entry : *cls->GetListOfDataMembers()) {
      auto* member = static_cast<TDataMember*>(entry);
      if (!(member->Property() & kIsPublic) || (member->Property() & kIsStatic))
        continue;
      const std::string memberType = pointee(strip(member->GetTrueTypeName()));
      if (member->GetArrayDim() > 0 || isScalar(memberType))
        continue;
      if (hasDictionary(memberType))
        collect(memberType, order, seen);
    }

    // And what its public methods hand back.  A format that keeps its storage
    // private -- which the pixel ones do -- reaches everything through
    // accessors, so following members alone stops at the first wrapper and the
    // methods that return the inner classes are then skipped as unbindable.
    for (auto* entry : *cls->GetListOfMethods()) {
      auto* method = static_cast<TMethod*>(entry);
      if (!(method->Property() & kIsPublic))
        continue;
      const std::string name = method->GetName();
      if (name.rfind("operator", 0) == 0 || name[0] == '~' || name == unqualified(cls->GetName()) ||
          name == cls->GetName())
        continue;
      const std::string result = pointee(strip(method->GetReturnTypeNormalizedName()));
      if (result.empty() || result == "void" || result == "auto" || isScalar(result))
        continue;
      if (!spanElement(strip(method->GetReturnTypeNormalizedName())).empty())
        continue;
      if (hasDictionary(result))
        collect(result, order, seen);
    }
    order.push_back(type);
  }

  /// The arguments of a public constructor Python could supply, or {}.
  ///
  /// A scalar, a column (a span of scalars) or another class that has been
  /// bound: those are the things a Python module has in its hands.  A raw
  /// pointer is not one of them, which is why a format that wants to be built
  /// from Python has to say so in its interface.
  std::vector<std::string> constructorArguments(TClass* cls, const std::set<std::string>& seen) {
    if (!cls || !cls->GetListOfMethods())
      return {};
    std::vector<std::string> best;
    for (auto* entry : *cls->GetListOfMethods()) {
      auto* method = static_cast<TMethod*>(entry);
      if (!(method->Property() & kIsPublic))
        continue;
      if (std::string(method->GetName()) != unqualified(cls->GetName()) &&
          std::string(method->GetName()) != cls->GetName())
        continue;
      if (!method->GetListOfMethodArgs() || method->GetListOfMethodArgs()->GetSize() == 0)
        continue;
      std::vector<std::string> arguments;
      bool usable = true;
      bool wiring = false;
      for (auto* a : *method->GetListOfMethodArgs()) {
        const std::string full = pointee(strip(static_cast<TMethodArg*>(a)->GetFullTypeName()));
        // A copy or move constructor is not a way to build one from Python.
        if (full == cls->GetName()) {
          usable = false;
          break;
        }
        if (!spanElement(full).empty() || (hasDictionary(full) && seen.count(full))) {
          wiring = true;
          arguments.push_back(full);
        } else if (isScalar(full)) {
          arguments.push_back(full);
        } else {
          usable = false;
          break;
        }
      }
      // At least one argument has to be another product or a column.  A
      // constructor taking only numbers is sizing the product, which
      // allocate() and a fill already do; the constructors that matter here
      // wire one product to another.
      if (usable && wiring && arguments.size() > best.size())
        best = std::move(arguments);
    }
    return best;
  }

}  // namespace

int generate_bindings(const char* library, const char* output, const char* productList) {
  if (gSystem->Load(library) < 0) {
    fprintf(stderr, "error: cannot load %s\n", library);
    return 1;
  }

  const std::vector<std::string> products = split(productList);

  std::vector<std::string> order;
  std::set<std::string> seen;
  for (const auto& product : products) {
    // A scalar product has no class to reflect: nanobind's own casters carry it
    // across by value, exactly as they already carry a class's scalar members.
    // ROOT has no TClass for a fundamental type, so asking for one is an error
    // rather than an empty answer.
    if (isScalar(product))
      continue;
    if (!TClass::GetClass(product.c_str())) {
      fprintf(stderr, "error: no dictionary for product %s\n", product.c_str());
      return 1;
    }
    collect(product, order, seen);
  }

  std::vector<std::string> body;
  std::vector<std::string> warnings;
  // Per class, its public non-static data members when every one of them is a
  // scalar.  That is what emplace() needs to build an aggregate from values.
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> aggregates;
  int totalFields = 0, totalViews = 0, totalMethods = 0, totalSkipped = 0;

  // Two classes can want the same Python name -- SiPixelDigisSoA::DeviceConstView
  // and SiPixelClustersSoA::DeviceConstView do -- and the second registration
  // would quietly shadow the first.  Where that happens, everything colliding
  // keeps its enclosing scope.
  std::map<std::string, int> nameUses;
  for (const auto& type : order)
    ++nameUses[pythonName(type)];
  std::map<std::string, std::string> uniqueName;
  for (const auto& type : order)
    uniqueName[type] = nameUses[pythonName(type)] > 1 ? sanitise(type) : pythonName(type);

  for (size_t index = 0; index < order.size(); ++index) {
    const std::string& type = order[index];
    TClass* cls = TClass::GetClass(type.c_str());
    const std::string variable = Form("c%zu", index);
    const std::string name = uniqueName.at(type);
    std::vector<std::string> lines, skipped;
    std::vector<std::pair<std::string, std::string>> members;  // (type, name), scalars only
    bool allMembersScalar = true;

    if (const std::string element = vectorElement(type); !element.empty()) {
      body.push_back(Form("  // ---- %s ----", type.c_str()));
      body.push_back(
          Form("  auto %s = nb::class_<%s>(m, \"%s\");", variable.c_str(), type.c_str(), name.c_str()));
      body.push_back(Form("  %s.def(\"size\", [](%s& o) { return o.size(); });", variable.c_str(), type.c_str()));
      body.push_back(Form("  %s.def(\"resize\", [](%s& o, size_t n) { o.resize(n); }, nb::arg(\"n\"));",
                          variable.c_str(),
                          type.c_str()));
      body.push_back(Form("  %s.def_prop_ro(\"data\", [](nb::handle h) {\n"
                          "    auto& o = nb::cast<%s&>(h);\n"
                          "    const size_t shape[1] = {o.size()};\n"
                          "    return nb::ndarray<nb::numpy, %s, nb::ndim<1>, nb::c_contig>(o.data(), 1, shape, h);\n"
                          "  });",
                          variable.c_str(),
                          type.c_str(),
                          element.c_str()));
      body.push_back("");
      totalViews += 1;
      totalMethods += 2;
      continue;
    }

    for (auto* entry : *cls->GetListOfDataMembers()) {
      auto* member = static_cast<TDataMember*>(entry);
      const std::string field = member->GetName();
      const std::string memberType = strip(member->GetTrueTypeName());
      const bool isPublic = (member->Property() & kIsPublic) != 0;
      const bool isStatic = (member->Property() & kIsStatic) != 0;

      if (!isPublic) {
        skipped.push_back(Form("field %s: not public", field.c_str()));
        continue;
      }

      if (isStatic) {
        if (!isScalar(memberType)) {
          skipped.push_back(Form("static %s: type %s", field.c_str(), memberType.c_str()));
          continue;
        }
        lines.push_back(Form("  %s.def_prop_ro_static(\"%s\", [](nb::handle) { return %s::%s; });",
                             variable.c_str(),
                             field.c_str(),
                             type.c_str(),
                             field.c_str()));
        ++totalFields;
        continue;
      }

      // A fixed-size array of scalars becomes a zero-copy numpy view.  The
      // extent comes from the dictionary, which is the whole reason this can be
      // generated: `float zv[1024]` reports its element type and its 1024.
      if (member->GetArrayDim() > 0) {
        allMembersScalar = false;
        if (member->GetArrayDim() != 1) {
          skipped.push_back(Form("field %s: %d-dimensional array", field.c_str(), member->GetArrayDim()));
          continue;
        }
        if (!isScalar(memberType)) {
          skipped.push_back(
              Form("field %s: array of %s, which is not a scalar", field.c_str(), memberType.c_str()));
          continue;
        }
        lines.push_back(Form(
            "  %s.def_prop_ro(\"%s\", [](nb::handle h) {\n"
            "    auto& o = nb::cast<%s&>(h);\n"
            "    const size_t shape[1] = {%d};\n"
            "    return nb::ndarray<nb::numpy, %s, nb::ndim<1>, nb::c_contig>(o.%s, 1, shape, h);\n"
            "  });",
            variable.c_str(),
            field.c_str(),
            type.c_str(),
            member->GetMaxIndex(0),
            memberType.c_str(),
            field.c_str()));
        ++totalViews;
        continue;
      }

      if (isScalar(memberType)) {
        members.emplace_back(memberType, field);
        lines.push_back(Form("  %s.def_rw(\"%s\", &%s::%s);",
                             variable.c_str(),
                             field.c_str(),
                             type.c_str(),
                             field.c_str()));
        ++totalFields;
        continue;
      }

      allMembersScalar = false;

      // A class member is handed back by reference, and the class itself has
      // been registered already: collect() ordered them deepest first.
      if (hasDictionary(memberType) && seen.count(memberType)) {
        lines.push_back(Form("  %s.def_prop_ro(\"%s\", [](nb::handle h) {\n"
                             "    auto& o = nb::cast<%s&>(h);\n"
                             "    return nb::cast(&o.%s, nb::rv_policy::reference_internal, h);\n"
                             "  });",
                             variable.c_str(),
                             field.c_str(),
                             type.c_str(),
                             field.c_str()));
        ++totalFields;
        continue;
      }

      skipped.push_back(Form("field %s: type %s has no dictionary", field.c_str(), memberType.c_str()));
    }

    if (allMembersScalar && !members.empty())
      aggregates.emplace(type, std::move(members));

    // Methods, as lambdas so the compiler deduces the return type.
    const std::string bare = pythonName(type);
    std::set<std::string> emitted;
    for (auto* entry : *cls->GetListOfMethods()) {
      auto* method = static_cast<TMethod*>(entry);
      if (!(method->Property() & kIsPublic))
        continue;
      const std::string method_name = method->GetName();
      if (method_name.rfind("operator", 0) == 0 || method_name[0] == '~')
        continue;
      if (method_name == cls->GetName() || method_name == unqualified(cls->GetName()))
        continue;  // constructor

      // A return type of `auto` is one the dictionary cannot spell -- typically
      // a wrapper's get().  Emitting a lambda rather than a member pointer
      // means the compiler deduces it, so these are usable after all; they come
      // back by reference, tied to the object they came from.
      const std::string result = strip(method->GetReturnTypeNormalizedName());
      const std::string span = spanElement(result);
      const bool deduced = result == "auto*" || result == "auto&" || result == "auto";
      const std::string returned = pointee(result);
      const bool classResult = hasDictionary(returned) && seen.count(returned);
      const bool usable =
          deduced || isScalar(result) || result == "void" || !span.empty() || classResult;
      if (!usable) {
        skipped.push_back(Form("method %s: returns %s", method_name.c_str(), result.c_str()));
        continue;
      }

      std::string parameters, arguments, argnames;
      bool ok = true;
      int argIndex = 0;
      if (method->GetListOfMethodArgs()) {
        for (auto* a : *method->GetListOfMethodArgs()) {
          auto* argument = static_cast<TMethodArg*>(a);
          const std::string argType = strip(argument->GetFullTypeName());
          if (!isScalar(argType)) {
            ok = false;
            skipped.push_back(
                Form("method %s: argument of type %s", method_name.c_str(), argType.c_str()));
            break;
          }
          const std::string argName = Form("a%d", argIndex++);
          parameters += Form(", %s %s", argType.c_str(), argName.c_str());
          arguments += (arguments.empty() ? "" : ", ") + argName;
          argnames += Form(", nb::arg(\"%s\")", argName.c_str());
        }
      }
      if (!ok)
        continue;

      // Const and non-const overloads of one name collapse to a single
      // binding; taking a member pointer here would be ambiguous, which is the
      // other reason these are lambdas.
      if (!emitted.insert(method_name).second)
        continue;

      // A column: the span says where it starts and how long it is, so the view
      // is exact without anything here knowing the product.
      if (!span.empty() && parameters.empty()) {
        lines.push_back(Form("  %s.def_prop_ro(\"%s\", [](nb::handle h) {\n"
                             "    auto& o = nb::cast<%s&>(h);\n"
                             "    const auto column = o.%s();\n"
                             "    const size_t shape[1] = {column.size()};\n"
                             "    return nb::ndarray<nb::numpy, %s, nb::ndim<1>, nb::c_contig>(\n"
                             "        const_cast<%s*>(column.data()), 1, shape, h);\n"
                             "  });",
                             variable.c_str(),
                             method_name.c_str(),
                             type.c_str(),
                             method_name.c_str(),
                             span.c_str(),
                             span.c_str()));
        ++totalViews;
        continue;
      }
      if (!span.empty()) {
        skipped.push_back(Form("method %s: returns a span but takes arguments", method_name.c_str()));
        continue;
      }

      const bool byReference = deduced || classResult;
      // -> decltype(auto), not plain auto: a lambda returning `o.method()`
      // deduces *by value*, so a method returning a reference would be bound to
      // a copy.  Reading one gives the right answer, which is why this went
      // unnoticed; taking its address does not, and a product built around such
      // an address points at a Python temporary that dies with it.
      lines.push_back(Form("  %s.def(\"%s\", [](%s& o%s) -> decltype(auto) { return o.%s(%s); }%s%s);",
                           variable.c_str(),
                           method_name.c_str(),
                           type.c_str(),
                           parameters.c_str(),
                           method_name.c_str(),
                           arguments.c_str(),
                           argnames.c_str(),
                           byReference ? ", nb::rv_policy::reference_internal" : ""));
      ++totalMethods;
    }

    body.push_back(Form("  // ---- %s ----", type.c_str()));
    for (const auto& reason : skipped) {
      body.push_back("  // skipped: " + reason);
      warnings.push_back(type + ": " + reason);
    }
    totalSkipped += skipped.size();
    body.push_back(Form("  auto %s = nb::class_<%s>(m, \"%s\");", variable.c_str(), type.c_str(), name.c_str()));
    for (const auto& line : lines)
      body.push_back(line);
    body.push_back("");
  }

  // How a product is created when a Python module asks the Event for one.  A
  // wrapper holding a unique_ptr is not usable default-constructed, and the
  // dictionary says so: it has a constructor taking unique_ptr<X>&&.
  std::vector<std::string> makers;
  for (const auto& product : products) {
    // A scalar is emplaced from the value Python passes in, so it needs no
    // Maker -- and has no TClass to ask about one.
    if (isScalar(product))
      continue;
    TClass* cls = TClass::GetClass(product.c_str());
    const std::string payload = ownedPayload(cls);
    if (payload.empty()) {
      // The primary template already says T{}, and saying it again here would
      // demand a default constructor of every product, including the ones that
      // reach Python from the EventSetup and are never allocated.
    } else {
      makers.push_back(Form("template <> struct Maker<%s> {\n"
                            "    static %s make() { return %s(std::make_unique<%s>()); }\n"
                            "  };",
                            product.c_str(),
                            product.c_str(),
                            product.c_str(),
                            payload.c_str()));
    }
  }

  std::ofstream out(output);
  out << "// Generated by tools/generate_bindings.C -- do not edit.\n"
      << "//\n"
      << "// Read out of the ROOT dictionaries produced by rootcling, so this file\n"
      << "// cannot drift from the C++ declarations: the build regenerates it whenever\n"
      << "// a product header changes.\n\n"
      << "#include <memory>\n"
      << "#include <type_traits>\n"
      << "#include <utility>\n"
      << "#include <stdexcept>\n"
      << "#include <string>\n"
      << "#include <typeindex>\n"
      << "#include <vector>\n\n"
      << "#include <nanobind/nanobind.h>\n"
      << "#include <nanobind/ndarray.h>\n\n"
      << "#include \"DataFormats/Products.h\"\n"
      << "#include \"Framework/Event.h\"\n"
      << "#include \"Framework/EDGetToken.h\"\n"
      << "#include \"Framework/EDPutToken.h\"\n"
      << "#include \"Framework/EventSetup.h\"\n"
      << "#include \"Framework/PayloadTypes.h\"\n\n"
      << "namespace nb = nanobind;\n\n"
      << "namespace {\n"
      << "  template <typename T>\n"
      << "  struct Maker {\n"
      << "    static T make() { return T{}; }\n"
      << "  };\n\n"
      << "  /// allocate(): construct the product in the Event and hand back a reference\n"
      << "  /// to fill.  A scalar has no storage to fill.\n"
      << "  template <typename T>\n"
      << "  nb::object allocateProduct(nb::handle self, edm::Event& event, edm::EDPutToken const& token) {\n"
      << "    if constexpr (std::is_arithmetic_v<T>) {\n"
      << "      throw nb::type_error(\n"
      << "          (token.typeName() + \" has no storage to fill: put(token, value), not allocate(token)\")\n"
      << "              .c_str());\n"
      << "    } else if constexpr (std::is_default_constructible_v<T>) {\n"
      << "      return nb::cast(&event.emplaceByIndex<T>(token.index(), Maker<T>::make()),\n"
      << "                      nb::rv_policy::reference_internal, self);\n"
      << "    } else {\n"
      << "      throw nb::type_error(\n"
      << "          (token.typeName() + \" cannot be default-constructed, so there is nothing to \"\n"
      << "                               \"allocate; emplace(token, ...) builds one instead\")\n"
      << "              .c_str());\n"
      << "    }\n"
      << "  }\n\n"
      << "  /// put(): copy the value into the Event.  A product that cannot be copied says\n"
      << "  /// so at run time; the branch that would not compile for it is discarded, which\n"
      << "  /// only a template guarantees.\n"
      << "  template <typename T>\n"
      << "  nb::object putProduct(edm::Event& event, edm::EDPutToken const& token, nb::handle value) {\n"
      << "    const auto wrongType = [&] {\n"
      << "      return nb::type_error((\"put() expects a value of type \" + token.typeName()).c_str());\n"
      << "    };\n"
      << "    if constexpr (std::is_arithmetic_v<T>) {\n"
      << "      try {\n"
      << "        event.emplaceByIndex<T>(token.index(), nb::cast<T>(value));\n"
      << "      } catch (nb::cast_error const&) {\n"
      << "        throw wrongType();\n"
      << "      }\n"
      << "      return nb::none();\n"
      << "    } else if constexpr (std::is_copy_constructible_v<T>) {\n"
      << "      try {\n"
      << "        event.emplaceByIndex<T>(token.index(), nb::cast<T const&>(value));\n"
      << "      } catch (nb::cast_error const&) {\n"
      << "        throw wrongType();\n"
      << "      }\n"
      << "      return nb::none();\n"
      << "    } else {\n"
      << "      throw nb::type_error(\n"
      << "          (token.typeName() + \" cannot be copied: allocate(token) and fill it in place\").c_str());\n"
      << "    }\n"
      << "  }\n\n";
  for (const auto& maker : makers)
    out << "  " << maker << "\n\n";
  out << "}  // namespace\n\n"
      << "void registerReflectedBindings(nb::module_& m) {\n";
  for (const auto& line : body)
    out << line << "\n";
  out << "}\n\n";

  // The Event's get/emplace dispatch, generated from the same product list, so
  // there is no table anywhere that has to be kept in step by hand.
  out << "nb::object reflectedGet(nb::handle self, edm::Event& event, edm::EDGetToken const& token) {\n";
  for (const auto& product : products)
    if (isScalar(product))
      out << "  if (token.typeName() == \"" << product << "\")\n"
          << "    return nb::cast(event.getByIndex<" << product << ">(token.index()));\n";
    else
      out << "  if (token.typeName() == \"" << product << "\")\n"
          << "    return nb::cast(&const_cast<" << product << "&>(event.getByIndex<" << product
          << ">(token.index())), nb::rv_policy::reference_internal, self);\n";
  out << "  throw nb::type_error((\"no bindings for product type '\" + token.typeName() + \"'\").c_str());\n"
      << "}\n\n";

  out << "nb::object reflectedAllocate(nb::handle self, edm::Event& event, edm::EDPutToken const& token,\n"
      << "                              nb::args args) {\n";
  for (const auto& product : products) {
    const auto arguments = constructorArguments(TClass::GetClass(product.c_str()), seen);
    if (arguments.empty()) {
      out << "  if (token.typeName() == \"" << product << "\")\n"
          << "    return allocateProduct<" << product << ">(self, event, token);\n";
      continue;
    }
    // A product that cannot be default-constructed but has a public
    // constructor Python can call: the arguments size and wire it, and what
    // comes back is still empty storage to fill.
    out << "  if (token.typeName() == \"" << product << "\") {\n"
        << "    if (args.size() == 0)\n"
        << "      return allocateProduct<" << product << ">(self, event, token);\n"
        << "    if (args.size() != " << arguments.size() << ")\n"
        << "      throw nb::type_error(\"allocate() of " << product << " takes no arguments or "
        << arguments.size() << "\");\n";
    std::string values;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const std::string element = spanElement(arguments[i]);
      if (!element.empty()) {
        out << "    auto column" << i << " = nb::cast<nb::ndarray<const " << element
            << ", nb::ndim<1>, nb::c_contig>>(args[" << i << "]);\n";
        // ROOT drops the namespace from std::span; anything else is spelled
        // the way the header spells it.
        const std::string spanType =
            arguments[i].rfind("span<", 0) == 0 ? "std::" + arguments[i] : arguments[i];
        values += (i ? ",\n                                  " : "") + spanType + "(column" + std::to_string(i) +
                  ".data(), column" + std::to_string(i) + ".size())";
      } else if (isScalar(arguments[i])) {
        values += (i ? ",\n                                  " : "") + std::string("nb::cast<") + arguments[i] +
                  ">(args[" + std::to_string(i) + "])";
      } else {
        values += (i ? ",\n                                  " : "") + std::string("nb::cast<") + arguments[i] +
                  "&>(args[" + std::to_string(i) + "])";
      }
    }
    out << "    return nb::cast(&event.emplaceByIndex<" << product << ">(token.index(), " << product << "(" << values
        << ")),\n"
        << "                    nb::rv_policy::reference_internal, self);\n"
        << "  }\n";
  }
  out << "  throw nb::type_error((\"no bindings for product type '\" + token.typeName() + \"'\").c_str());\n"
      << "}\n\n";

  out << "nb::object reflectedPut(edm::Event& event, edm::EDPutToken const& token, nb::handle value) {\n";
  for (const auto& product : products)
    out << "  if (token.typeName() == \"" << product << "\")\n"
        << "    return putProduct<" << product << ">(event, token, value);\n";
  out << "  throw nb::type_error((\"no bindings for product type '\" + token.typeName() + \"'\").c_str());\n"
      << "}\n\n";

  // emplace(): build the product from Python values and move it in.  Unlike
  // put(), which needs a C++ object of the product's type to copy, this is what
  // constructs one -- so a list becomes a vector here and nowhere else.  What
  // can be built is decided per product, from the dictionary: a scalar, a
  // vector of scalars, or an aggregate whose every member is a scalar.
  out << "nb::object reflectedEmplace(edm::Event& event, edm::EDPutToken const& token, nb::args args) {\n";
  for (const auto& product : products) {
    out << "  if (token.typeName() == \"" << product << "\") {\n";
    const std::string element = vectorElement(product);
    if (isScalar(product)) {
      out << "    if (args.size() != 1)\n"
          << "      throw nb::type_error(\"emplace() of " << product << " takes one value\");\n"
          << "    event.emplaceByIndex<" << product << ">(token.index(), nb::cast<" << product
          << ">(args[0]));\n";
    } else if (!element.empty()) {
      out << "    if (args.size() != 1)\n"
          << "      throw nb::type_error(\"emplace() of " << product << " takes one sequence\");\n"
          << "    " << product << " built;\n"
          << "    for (nb::handle item : args[0])\n"
          << "      built.push_back(nb::cast<" << element << ">(item));\n"
          << "    event.emplaceByIndex<" << product << ">(token.index(), std::move(built));\n";
    } else if (const auto found = aggregates.find(product); found != aggregates.end()) {
      const auto& members = found->second;
      std::string names, values;
      for (std::size_t i = 0; i < members.size(); ++i) {
        names += (i ? ", " : "") + members[i].second;
        values += (i ? ",\n                            " : "") + std::string("nb::cast<") + members[i].first +
                  ">(args[" + std::to_string(i) + "])";
      }
      out << "    static_assert(std::is_aggregate_v<" << product << ">,\n"
          << "                  \"" << product << " is no longer an aggregate\");\n"
          << "    if (args.size() != " << members.size() << ")\n"
          << "      throw nb::type_error(\"emplace() of " << product << " takes " << members.size()
          << " values: " << names << "\");\n"
          << "    " << product << " built{" << values << "};\n"
          << "    event.emplaceByIndex<" << product << ">(token.index(), std::move(built));\n";
    } else {
      out << "    throw nb::type_error(\n"
          << "        \"" << product
          << " cannot be built from Python values: allocate(token) and fill it in place\");\n";
    }
    out << "    return nb::none();\n"
        << "  }\n";
  }
  out << "  throw nb::type_error((\"no bindings for product type '\" + token.typeName() + \"'\").c_str());\n"
      << "}\n\n";

  // The EventSetup is keyed by type and holds no tokens, so its dispatch takes
  // the name directly.  Same list again: a type is reflected once, whether it
  // reaches Python from the Event or from the EventSetup.
  out << "nb::object reflectedESGet(nb::handle self, edm::EventSetup const& eventSetup, std::string const& "
         "typeName) {\n";
  for (const auto& product : products)
    if (isScalar(product))
      out << "  if (typeName == \"" << product << "\")\n"
          << "    return nb::cast(eventSetup.get<" << product << ">());\n";
    else
      out << "  if (typeName == \"" << product << "\")\n"
          << "    return nb::cast(&const_cast<" << product << "&>(eventSetup.get<" << product
          << ">()), nb::rv_policy::reference_internal, self);\n";
  out << "  throw nb::type_error((\"no bindings for product type '\" + typeName + \"'\").c_str());\n"
      << "}\n\n";

  // The payload table, from the same list again.  A Python module names a
  // product type with a string and the registry keys on a std::type_index, so
  // something has to bridge the two; generating it here means the product list
  // is written once, in the backend's Makefile, rather than once there and
  // once more in a header that nothing checks against it.
  out << "namespace edm {\n"
      << "  std::vector<std::string> const& payloadTypeNames() {\n"
      << "    static std::vector<std::string> const names = {";
  for (std::size_t i = 0; i < products.size(); ++i)
    out << (i ? ", " : "") << "\"" << products[i] << "\"";
  out << "};\n"
      << "    return names;\n"
      << "  }\n\n"
      << "  std::type_index payloadTypeIndex(std::string const& name) {\n";
  for (const auto& product : products)
    out << "    if (name == \"" << product << "\")\n"
        << "      return std::type_index(typeid(" << product << "));\n";
  out << "    std::string known;\n"
      << "    for (auto const& candidate : payloadTypeNames()) {\n"
      << "      if (not known.empty())\n"
      << "        known += \", \";\n"
      << "      known += candidate;\n"
      << "    }\n"
      << "    throw std::runtime_error(\"unsupported payload type '\" + name + \"'; known types are \" + known);\n"
      << "  }\n"
      << "}  // namespace edm\n";

  for (const auto& warning : warnings)
    fprintf(stderr, "  warning: %s\n", warning.c_str());
  printf("  %zu class(es), %d field(s), %d view(s), %d method(s), %d skipped\n",
         order.size(),
         totalFields,
         totalViews,
         totalMethods,
         totalSkipped);
  return 0;
}
