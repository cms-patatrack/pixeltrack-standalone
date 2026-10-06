# Scan — 1 to 190 threads

Run on 6 October 2026.  48 runs, no failures.
Discussed in [README.md](README.md), D30.

## What was run

Two configurations of the same build, on a dual-socket AMD EPYC 9655 (Zen 5, 2 × 96 cores):

| config | configuration |
|---|---|
| `cxx` | `reco.ini`, every module in C++ |
| `python` | `reco-optimised.ini` |

The build uses the flags of the optimised master,
`-O3 -march=native -ffp-contract=off`, and free-threaded CPython 3.14 with the
GIL disabled.

## How

1 to 190 threads, streams = threads, with N threads pinned with `taskset` to
the first N physical cores: cores 0–94 of socket 0 up to 95 threads, then
cores 96–190 of socket 1; SMT siblings never used.  Each run processes
max(10000, 500 × threads) events after max(1000, 10 × threads) of warm up,
the same protocol as the scan of the serial backend in `doc/serial`.  One run
per point; the run-to-run spread of this protocol is 1–3% up to 64 threads.

## Files

| file | contents |
|---|---|
| `scan.csv` | all 48 measurements |
| `throughput.png` | throughput against threads |
| `ratio.png` | Python / C++ |
| `resources/<config>-<threads>.json` | the time spent in each module (`--resources`), for `reco` and `reco-optimised` at 1, 8, 32, 95 and 190 threads, measured separately with the same protocol |
