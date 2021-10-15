#!/usr/bin/env python3

import json
import argparse
import statistics
import collections

cpuCoresSet = None

def analyzeList(names, inputData, peakName):
    data = {x:[] for x in names}
    for meas in inputData[1:-1]: # skip first and last
        for key, lst in data.items():
            if key in meas:
                lst.append(float(meas[key]))
    peak = 0
    for key, lst in data.items():
        if len(lst) > 0:
            if key == peakName:
                peak = max(lst)
            print(" {}: mean {:.3f} stdev {:3f} min {:3f} max {:3f}".format(key, statistics.mean(lst),
                                                                            statistics.stdev(lst) if len(lst) > 1 else 0,
                                                                            min(lst), max(lst)))
    return peak

def measurementKey(meas):
    if "streams" in meas:
        return meas["streams"]
    return tuple(m["streams"] for m in meas["programs"])

def hostMemory(data, results):
    for res in data["results"]:
        if not "monitor" in res:
            continue
        mon = res["monitor"]
        if "host" in mon:
            if "process" in mon["host"]:
                print("Host:")
                results[measurementKey(res)]["rss"].append(analyzeList(["rss"], mon["host"]["process"], "rss"))
            elif "processes" in mon["host"]:
                procs = mon["host"]["processes"]
                mem = []
                for i in range(1, len(procs[0])-1):
                    s = 0
                    for p in procs:
                        s += p[i]["rss"]
                    mem.append(dict(rss=s))
                results[measurementKey(res)]["rss"].append(analyzeList(["rss"], mem, "rss"))

def hostClock(data, results):
    for res in data["results"]:
        if not "monitor" in res:
            continue
        mon = res["monitor"]
        if "host" in mon:
            if "cpu" in mon["host"]:
                for core in mon["host"]["cpu"]:
                    if cpuCoresSet is None or int(core) in cpuCoresSet:
                        print(core)
                        results[measurementKey(res)]["cpuClock {}".format(core)].append(analyzeList(["clock"], mon["host"]["cpu"][core], "clock"))

def cudaStats(data, results):
    for res in data["results"]:
        if not "monitor" in res:
            continue
        mon = res["monitor"]
        if "cuda" in mon:
            for dev in mon["cuda"]:
                print("CUDA {}".format(dev))
                results[measurementKey(res)]["gpuMem"].append(analyzeList(["temperature","clock","proc_mem_use"], mon["cuda"][dev], "proc_mem_use"))

def throughput(data, results):
    for res in data["results"]:
        results[measurementKey(res)]["throughput"].append(res["throughput"])

def main(opts):
    with open(opts.file) as inp:
        data = json.load(inp)

    results = collections.defaultdict(lambda: collections.defaultdict(list))
    throughput(data, results)
    hostMemory(data, results)
#    hostClock(data, results)
    cudaStats(data, results)
    print()
    for st, res in results.items():
        print("Streams {}".format(st))
        for key, values in res.items():
            print(" {}: mean {:.3f} stdev {:3f} min {:3f} max {:3f}".format(key, statistics.mean(values),
                                                                            statistics.stdev(values) if len(values) > 1 else 0,
                                                                            min(values), max(values)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze monitoring data in result JSON files")
    parser.add_argument("file", type=str,
                        help="Path to JSON file to analyze")
    opts = parser.parse_args()

    main(opts)
