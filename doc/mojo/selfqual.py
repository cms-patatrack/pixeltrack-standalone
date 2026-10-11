#!/usr/bin/env python3
"""Apply the Mojo compiler's own "use 'Self.X' instead" fixits, iteratively.

The 0.26.2 dialect requires struct parameters to be referenced as `Self.T`
inside the struct body, while generic *function* parameters must stay bare.
Regex cannot tell the two apart, but the compiler can: it reports every site
that needs qualifying with an exact line:column. This drives a
compile -> apply only the flagged (line, col) -> recompile loop, which is
exact by construction and can never touch a function parameter.

Several rounds are normally needed: a parse error stops elaboration, so each
round uncovers more sites.

Usage:
    python3 selfqual.py <comma-separated target .mojo files> <driver.mojo> <cwd>

`driver.mojo` should *use* symbols from the targets, not merely import them --
a bare import elaborates almost nothing and will report a false clean.
"""
import os
import re
import subprocess
import sys

TARGETS = [os.path.realpath(p) for p in sys.argv[1].split(",")]
DRIVER = sys.argv[2]
CWD = sys.argv[3]
SCRATCH = os.path.dirname(DRIVER)

PAT = re.compile(
    r"^(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+): error: unqualified access to "
    r"struct parameter '(?P<name>[^']+)'; use '(?P<fix>[^']+)' instead$"
)


def compile_once():
    p = subprocess.run(
        ["pixi", "run", "mojo", "build", "--emit", "object", "-I", ".", DRIVER,
         "-o", os.path.join(SCRATCH, "drv.o")],
        cwd=CWD, capture_output=True, text=True)
    return p.stdout + p.stderr


total = 0
for round_no in range(1, 60):
    out = compile_once()
    hits = []
    for line in out.splitlines():
        m = PAT.match(line.strip())
        if m and os.path.realpath(m.group("file")) in TARGETS:
            hits.append((os.path.realpath(m.group("file")),
                         int(m.group("line")), int(m.group("col")),
                         m.group("name"), m.group("fix")))
    others = [l for l in out.splitlines()
              if " error: " in l and not PAT.match(l.strip())]
    if not hits:
        print(f"round {round_no}: no Self-qual fixits left ({total} applied total)")
        for l in others:
            print("  REMAINING:", l.strip())
        break

    srcs = {}
    # apply bottom-up so earlier lines/columns keep their offsets
    for (path, ln, col, name, fix) in sorted(set(hits), reverse=True):
        if path not in srcs:
            srcs[path] = open(path, encoding="utf-8").read().split("\n")
        src = srcs[path]
        text = src[ln - 1]
        start = col - 1
        got = text[start:start + len(name)]
        if got != name:
            print(f"  SKIP {os.path.basename(path)}:{ln}:{col} "
                  f"expected {name!r} found {got!r}")
            continue
        src[ln - 1] = text[:start] + fix + text[start + len(name):]
        total += 1
    for path, src in srcs.items():
        open(path, "w", encoding="utf-8").write("\n".join(src))
    print(f"round {round_no}: applied {len(set(hits))} fixits; "
          f"{len(others)} other error line(s)")
else:
    print("gave up after 60 rounds")
