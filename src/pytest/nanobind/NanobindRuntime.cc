// nanobind ships a single-file amalgamation of its runtime, meant to be
// compiled into whichever binary embeds it.  It lives in site-packages rather
// than in this tree, so the backend makefile passes its path in.
//
// It is compiled into libFramework.so so that there is exactly one nanobind
// runtime in the process, shared by the executable and by every plugin: two
// copies would each keep their own type registry and casts would fail across
// the boundary.
// nanobind's amalgamation is third-party code and trips this repository's
// -Werror=strict-aliasing.  The backend makefile bakes its flags in when the
// build templates are eval'd, so a target-specific override cannot reach this
// object; the suppression goes here instead, where it is at least visible.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#include EDM_NB_COMBINED
#pragma GCC diagnostic pop
