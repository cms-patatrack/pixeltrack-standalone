# Scan — the Python modules with namespaces

Run on 6 October 2026.  30 runs, no failures.
Discussed in [README.md](README.md), D31.

## What was run

The same build and protocol as [scan.md](scan.md), with the Python modules
importing numpy and `pixel_clusters_common` through `namespaces.py`:

| config | Python modules | configuration | threads |
|---|---|---|---|
| `python-namespaces` | with namespaces | `reco-optimised.ini` | the 24 points of `scan.md` |
| `cxx-recheck` | — | `reco.ini` | 1, 32, 190 |
| `python-recheck` | before, through `EDM_PYTHON_DIR` | `reco-optimised.ini` | 1, 32, 190 |

The two recheck configurations repeat a few points of `scan.csv` with the same
binary, to show that the machine and the build did not change: they agree
with it within 0.6% for C++ and 1.8% for Python, so the C++ and the "before"
Python of the comparison are taken from `scan.csv`.

## Files

| file | contents |
|---|---|
| `scan-namespaces.csv` | all 30 measurements |
| `throughput-namespaces.png` | throughput against threads: C++, Python before and with namespaces |
