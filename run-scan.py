#!/usr/bin/env python3

import os
import re
import json
import time
import socket
import argparse
import statistics
import subprocess
import collections
import multiprocessing

# Make CUDA_VISIBLE_DEVICES order match to nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Number of events for each application
n_events_unit = 1000
n_blocks_per_stream = {
    "fwtest": 1,
    "cuda": {"": 100, "transfer": 100},
    "cudadev": {"": 100, "transfer": 100},
    "cudauvm": {"": 100, "transfer": 100},
    "cudacompat": {"": 8},
    "serial": {"": 8},
}

# 30 ev/s * 8 hours should the sufficent and fit into signed int for ~2k threads
background_events_per_thread = 30*3600*8

result_re = re.compile("Processed (?P<events>\d+) events in (?P<time>\S+) seconds, throughput (?P<throughput>\S+) events/s, CPU usage per thread: (?P<cpueff>\d+(.\d+)?)%")

Measurement = collections.namedtuple("Measurement", ["events", "time", "throughput", "cpueff"])
GPU = collections.namedtuple("GPU", ["id", "name", "driver_version"])
GPUStatus = collections.namedtuple("GPUStatus", ["utilization", "temperature", "power", "clock"])
BackgroundJob = collections.namedtuple("BackgroundJob", ["handle", "logfile", "cores"])


class Monitor:
    def __init__(self, opts, cudaDevices=[]):
        self._intervalSeconds = opts.monitorSeconds
        self._monitorMemory = opts.monitorMemory
        self._monitorClock = opts.monitorClock
        self._monitorUtilization = opts.monitorUtilization
        self._monitorCuda = opts.monitorCuda

        self._timeStamp = []
        self._dataProcess = []
        self._dataClock = {x: [] for x in range(0, multiprocessing.cpu_count())}
        self._dataCuda = {x: [] for x in cudaDevices}

    def setIntervalSeconds(self, interval):
        self._intervalSeconds = interval

    def intervalSeconds(self):
        return self._intervalSeconds

    def snapshot(self, pid=None, cudaDevices=[]):
        if self._intervalSeconds is None:
            return
        self._timeStamp.append(time.strftime("%y-%m-%d %H:%M:%S"))

        if self._monitorMemory or self._monitorUtilization:
            proc = dict()
            if self._monitorMemory:
                proc["rss"] = processRss(pid) if pid is not None else 0
            if self._monitorUtilization:
                proc["utilization"] = processUtilization(pid) if pid is not None else 0.0
            self._dataProcess.append(proc)
        if self._monitorClock:
            clocks = processClock()
            for key, lst in self._dataClock.items():
                lst.append(dict(clock=clocks.get(key, -1.0)))

        if self._monitorCuda:
            for dev in cudaDevices:
                data = {}
                data.update(cudaDeviceStatus(dev)._asdict())
                mem = cudaDeviceProcessMemory(dev, pid) if pid is not None else 0
                data["proc_mem_use"] = mem
                self._dataCuda[dev].append(data)

    def toArrays(self):
        data = {}
        if self._intervalSeconds is not None:
            data["time"] = self._timeStamp
            if self._monitorMemory or self._monitorUtilization or self._monitorClock:
                data["host"] = {}
                if self._monitorMemory or self._monitorUtilization:
                    data["host"]["process"] = self._dataProcess
                if self._monitorClock:
                    data["host"]["cpu"] = self._dataClock
            if self._monitorCuda:
                data["cuda"] = self._dataCuda
        return data


def printMessage(*args):
    print(time.strftime("%y-%m-%d %H:%M:%S"), *args)

def throughput(output, filename):
    for line in output:
        m = result_re.search(line)
        if m:
            printMessage(line.rstrip())
            return Measurement(int(m.group("events")), float(m.group("time")), float(m.group("throughput")), float(m.group("cpueff")))

    raise Exception("Did not find throughput from the log")

def partition_cores(cores, nth):
    if nth >= len(cores):
        return (cores, [])

    return (cores[0:nth], cores[nth:])

def listCudaDevices():
    try:
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    except FileNotFoundError:
        return {}
    output = p.communicate()[0]
    ret = {}
    for line in output.split("\n"):
        if line:
            s = line.split(",")
            gpu = GPU(*[x.strip().rstrip() for x in s])
            ret[gpu.id] = gpu
    return ret

def cudaDeviceStatus(dev):
    p = subprocess.Popen(["nvidia-smi", "--id="+dev, "--query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    output = p.communicate()[0]
    s = output.rstrip().split(",")
    def convert(s):
        try:
            return float(s)
        except ValueError:
            return s
    return GPUStatus(*[convert(x.strip().rstrip()) for x in s])

def cudaDeviceProcessMemory(dev, pid):
    """In MB"""
    p = subprocess.Popen(["nvidia-smi", "--id="+dev, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    output = p.communicate()[0]
    for line in output.split("\n"):
        if line:
            s = line.split(",")
            if s[0].strip().rstrip() == str(pid):
                return float(s[1].strip().rstrip())
    return 0

def processRss(pid):
    """In MB"""
    # from https://stackoverflow.com/a/48397534
    with open("/proc/{}/status".format(pid)) as f:
        content = f.read()
    if not "VmRSS:" in content:
        return 0
    memusage = content.split('VmRSS:')[1].split('\n')[0][:-3]
    return float(memusage.strip())/1024.0

def processClock():
    """In MHz"""
    ret = {}
    with open("/proc/cpuinfo") as f:
        cpuId = -1
        for line in f:
            if "processor" in line:
                cpuid = int(int(line.split(":")[1]))
            elif "cpu MHz" in line:
                ret[cpuid] = float(line.split(":")[1])
                cpuid = -1
    return ret

def processUtilization(pid):
    p = subprocess.Popen(["ps", "-p", str(pid), "-o", "%cpu", "--no-header"], stdout=subprocess.PIPE, universal_newlines=True)
    output = p.communicate()[0]
    return float(output)

def _run(processUntil, nstr, cores_main, opts, logfilename, monitor, cudaDevices=[]):
    nth = len(cores_main)
    with open(logfilename, "w") as logfile:
        taskset = []
        nvprof = []
        command = [opts.program] + processUntil + ["--numberOfStreams", str(nstr), "--numberOfThreads", str(nth)] + opts.args
        if opts.taskset:
            taskset = ["taskset", "-c", ",".join(cores_main)]

        logfile.write(" ".join(taskset+command))
        logfile.write("\n----\n")
        logfile.flush()
        if opts.dryRun:
            print(" ".join(taskset+command))
            return (0, 0)
        env = None
        if len(cudaDevices) > 0:
            visibleDevices = ",".join(opts.cudaDevices)
            logfile.write("export CUDA_DEVICE_ORDER=PCI_BUS_ID\n")
            logfile.write("export CUDA_VISIBLE_DEVICES="+visibleDevices+"\n")
            logfile.flush()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visibleDevices)
        p = subprocess.Popen(taskset+command, stdout=logfile, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
        monitor.snapshot(pid=p.pid, cudaDevices=cudaDevices)
        while True:
            try:
                p.wait(timeout=monitor.intervalSeconds())
                monitor.snapshot(cudaDevices=cudaDevices)
            except subprocess.TimeoutExpired:
                monitor.snapshot(pid=p.pid, cudaDevices=cudaDevices)
                continue
            except KeyboardInterrupt:
                try:
                    p.terminate()
                except OSError:
                    pass
                p.wait()
            break
        if p.returncode != 0:
            raise Exception("Got return code %d, see output in the log file %s" % (p.returncode, logfilename))
    with open(logfilename) as logfile:
        return throughput(logfile, logfilename)

def runEvents(nev, *args, **kwargs):
    return _run(["--maxEvents", str(nev)], *args, **kwargs)

def runMinutes(mins, *args, **kwargs):
    return _run(["--runForMinutes", str(mins)], *args, **kwargs)

def launchBackground(opts, cores_bkg, logfilepattern):
    if opts.fill <= 0:
        return []
    nth = len(cores_bkg)
    if nth == 0:
        return []
    nev = background_events_per_thread * nth
    taskset = []
    exe = os.path.join(os.path.dirname(opts.program), "serial")

    serials = []
    nth_per_process = nth
    if opts.bkgThreads > 0 and opts.bkgThreads < nth:
        nth_per_process = opts.bkgThreads
    nprocesses = nth // nth_per_process
    if nth % nth_per_process != 0:
        nprocesses += 1
    for ibkg in range(0, nprocesses):
        logfile = open(logfilepattern.format(ibkg), "w")
        cores = cores_bkg[ibkg*nth_per_process:(ibkg+1)*nth_per_process]
        nth_this = len(cores)

        command = [exe, "--maxEvents", str(nev), "--numberOfThreads", str(nth_this)]
        if opts.taskset:
            taskset = ["taskset", "-c", ",".join(cores)]
        if opts.bkgNice is not None:
            taskset.extend(["nice", "-n", str(opts.bkgNice)])

        logfile.write(" ".join(taskset+command))
        logfile.write("\n----\n")
        logfile.flush()
        if opts.dryRun:
            print(" ".join(taskset+command))
            continue
        serials.append(BackgroundJob(subprocess.Popen(taskset+command, stdout=logfile, stderr=subprocess.STDOUT, universal_newlines=True),
                                     logfile, cores))
    return serials

def getEventsPerStream(program, opts):
    ret = opts.eventsPerStream
    if ret is None and opts.runForMinutes < 0:
        tmp = n_blocks_per_stream.get(os.path.basename(program), None)
        if tmp is None:
            raise Exception("No default number of event blocks for program %s, and --eventsPerStream was not given" % program)
        if isinstance(tmp, dict):
            if "--transfer" in opts.args:
                eventBlocksPerStream = tmp["transfer"]
            else:
                eventBlocksPerStream = tmp[""]
        else:
            eventBlocksPerStream = tmp
        return eventBlocksPerStream * n_events_unit
    return ret

def main(opts):
    ncores = multiprocessing.cpu_count()
    if opts.fill > 0:
        ncores = opts.fill

    cudaDevices = listCudaDevices()
    print("Found {} devices".format(len(cudaDevices)))
    for i, d in cudaDevices.items():
        print(" {} {} driver {}".format(i, d.name, d.driver_version))

    if len(opts.tasksetCores) > 0:
        cores = opts.tasksetCores[:]
    else:
        cores = [str(x) for x in range(0, ncores)]
    maxThreads = len(cores)
    if opts.maxThreads > 0:
        maxThreads = min(maxThreads, opts.maxThreads)

    nthreads = range(opts.minThreads,maxThreads+1)
    if len(opts.numThreads) > 0:
        nthreads = [x for x in opts.numThreads if x >= opts.minThreads and x <= maxThreads]
    n_streams_threads = [(i, i) for i in nthreads]
    if len(opts.numStreams) > 0:
        n_streams_threads = [(s, t) for t in nthreads for s in opts.numStreams]

    nev_per_stream = getEventsPerStream(opts.program, opts)

    data = dict(
        program=opts.program,
        args=" ".join(opts.args),
        results=[]
    )
    outputJson = opts.output+".json"
    alreadyExists = set()
    if not opts.overwrite and os.path.exists(outputJson):
        with open(outputJson) as inp:
            data = json.load(inp)
    if not opts.append:
        for res in data["results"]:
            alreadyExists.add( (res["streams"], res["threads"]) )

    hostname = socket.gethostname()
    stop = False

    for nstr, nth in n_streams_threads:
        if nstr == 0:
            nstr = nth
        if (nstr, nth) in alreadyExists:
            continue

        (cores_main, cores_bkg) = partition_cores(cores, nth)

        mins = -1
        nev = -1
        if opts.runForMinutes >= 0:
            mins = opts.runForMinutes
            def run(postfix, **kwargs): return runMinutes(mins, nstr, cores_main, opts, opts.output+postfix, **kwargs)
        else:
            if opts.maxStreamsToAddEvents > 0 and nstr > opts.maxStreamsToAddEvents:
                nev = nev_per_stream * opts.maxStreamsToAddEvents
            else:
                nev = nev_per_stream*nstr
            def run(postfix, **kwargs): return runEvents(nev, nstr, cores_main, opts, opts.output+postfix, **kwargs)

        if opts.warmup:
          printMessage("Warming up")
          wmon = Monitor(opts)
          wmon.setIntervalSeconds(None)
          run("_warmup.txt", monitor=wmon, cudaDevices=opts.cudaDevices)
          print()
          opts.warmup = False

        backgroundJobs = launchBackground(opts, cores_bkg, opts.output+"_log_nstr{}_nth{}_bkg".format(nstr, nth)+"{}.txt")
        if len(backgroundJobs) > 0:
            msg = "Background serial\n"
            for job in backgroundJobs:
                msg += " pid {}".format(job.handle.pid)
                if opts.taskset:
                    msg +=", running on cores " + ",".join(job.cores)
                msg += "\n"
            printMessage(msg)

        try:
            msg = "Number of streams {} threads {}".format(nstr, nth)
            if nev >= 0:
                msg += " events {}".format(nev)
            else:
                msg += " minutes {}".format(mins)
            if opts.taskset:
                msg += ", running on cores " + ",".join(cores_main)
            if len(opts.cudaDevices) > 0:
                msg += ", running on devices " + ",".join(opts.cudaDevices)
            printMessage(msg)
            throughputs = []
            for i in range(opts.repeat):
                tryAgain = opts.tryAgain
                while tryAgain > 0:
                    try:
                        monitor = Monitor(opts, cudaDevices=opts.cudaDevices)
                        measurement = run("_log_nstr{}_nth{}_n{}.txt".format(nstr, nth, i), monitor=monitor, cudaDevices=opts.cudaDevices)
                        break
                    except Exception as e:
                        tryAgain -= 1
                        if tryAgain == 0:
                            raise
                        print("Got exception (see below), trying again ({} times left)".format(tryAgain))
                        print("--------------------")
                        print(str(e))
                        print("--------------------")

                if opts.dryRun:
                    continue
                throughputs.append(measurement.throughput)
                d = dict(
                    hostname=hostname,
                    threads=nth,
                    streams=nstr,
                    events=measurement.events,
                    throughput=measurement.throughput,
                    cpueff=measurement.cpueff,
                )
                if monitor.intervalSeconds() is not None:
                    d["monitor"]=monitor.toArrays()
                if len(opts.cudaDevices) > 0:
                    d["cudaDevices"] = {
                        x: dict(name=cudaDevices[x].name, driver_version=cudaDevices[x].driver_version) for x in opts.cudaDevices
                    }
                data["results"].append(d)
                # Save results after each test
                with open(outputJson, "w") as out:
                    json.dump(data, out, indent=2)
                if opts.stopAfterWallTime > 0 and measurement.time > opts.stopAfterWallTime:
                    stop = True
                    break
        finally:
            if len(backgroundJobs) > 0:
                printMessage("Run complete, terminating background serial jobs")
                try:
                    for job in backgroundJobs:
                        job.handle.terminate()
                except OSError:
                    pass
                for job in backgroundJobs:
                    job.handle.wait()
                    job.logfile.close()

        thr = 0
        stdev = 0
        if len(throughputs) > 0:
            thr = statistics.mean(throughputs)
            if len(throughputs) > 1:
                stdev = statistics.stdev(throughputs)
        printMessage("Number of streams {} threads {}, average throughput {} stdev {}".format(nstr, nth, thr, stdev))
        print()
        if stop:
            print("Reached max wall time of %d s, stopping scan" % opts.stopAfterWallTime)
            break


def addCommonArguments(parser):
    output_group = parser.add_argument_group("JSON output arguments")
    output_group.add_argument("-o", "--output", type=str, default="result",
                              help="Prefix of output JSON and log files. If the output JSON file exists, it will be updated (see also --overwrite) (default: 'result')")
    output_group.add_argument("--overwrite", action="store_true",
                              help="Overwrite the output JSON instead of updating it")
    output_group.add_argument("--append", action="store_true",
                              help="Append new (stream, threads) results insteads of ignoring already existing point")

    monitor_group = parser.add_argument_group("Monitoring arguments",
                                              description="These arguments can be used to enable various monitoring of the program being tested. The data is stored in the result JSON file.")
    monitor_group.add_argument("--monitorSeconds", type=int, default=-1,
                               help="Store monitoring data with intervals of this many seconds (default -1 for disabled)")
    monitor_group.add_argument("--monitorMemory", action="store_true",
                               help="Enable monitoring of host memory")
    monitor_group.add_argument("--monitorClock", action="store_true",
                               help="Enable monitoring of CPU core clocks")
    monitor_group.add_argument("--monitorUtilization", action="store_true",
                               help="Enable monitoring of CPU utilization with 'ps'")
    monitor_group.add_argument("--monitorCuda", action="store_true",
                               help="Enable monitoring of CUDA devices (utilization, power, memory etc)")

    parser.add_argument("--tryAgain", type=int, default=1,
                        help="In case of failure on a point, try again at most this many times (default: 1)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run the command once before starting the profiling")
    parser.add_argument("--dryRun", action="store_true",
                        help="Print out commands, don't actually run anything")

def parseCommonArguments(parser):
    opts = parser.parse_args()
    if opts.monitorSeconds < 0:
        opts.monitorSeconds = None

    return opts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Run a scan of a given test program.
Note that this program does not honor CUDA_VISIBLE_DEVICES, use --cudaDevices instead.
""")
    parser.add_argument("program", type=str,
                        help="Path to the test program to run.")

    addCommonArguments(parser)

    scan_group = parser.add_argument_group("Scan arguments")
    scan_group.add_argument("--repeat", type=int, default=1,
                            help="Repeat each point this many times (default: 1)")
    scan_group.add_argument("--minThreads", type=int, default=1,
                            help="Minimum number of threads to use in the scan (default: 1)")
    scan_group.add_argument("--maxThreads", type=int, default=-1,
                            help="Maximum number of threads to use in the scan (default: -1 for the number of cores)")
    scan_group.add_argument("--numThreads", type=str, default="",
                            help="Comma separated list of numbers of threads to use in the scan (default: empty for all)")
    scan_group.add_argument("--numStreams", type=str, default="",
                            help="Comma separated list of numbers of streams to use in the scan (default: empty for always the same as the number of threads). If both number of threads and number of streams have more than 1 element, a 2D scan is done with all the combinations")
    scan_group.add_argument("--stopAfterWallTime", type=int, default=-1,
                            help="Stop running after the wall time of the job reaches this many in seconds (default: -1 for no limit)")

    nevents_group = parser.add_argument_group("Setting number of events arguments")
    nevents_group.add_argument("--eventsPerStream", type=int, default=None,
                               help="Number of events to be used per EDM stream (default: 400*4kev for cuda, others also hardcoded in the top of the script file)")
    nevents_group.add_argument("--maxStreamsToAddEvents", type=int, default=-1,
                               help="Maximum number of streams to add events (default: -1 for no limit")
    nevents_group.add_argument("--runForMinutes", type=int, default=-1,
                               help="Process the set of events until this many minutes has elapsed. Conflicts with --eventsPerStream and --maxStreamsToAddEvents. (default -1 for disabled)")

    fill_group = parser.add_argument_group("Node filling and pinning arguments")
    fill_group.add_argument("--taskset", action="store_true",
                            help="Use taskset to explicitly set the cores where to run on")
    fill_group.add_argument("--tasksetCores", type=str, default="",
                            help="Comma-separated list of cores to be used for taskset in that order. Default (empty) is to use range(0, N(cores))")
    fill_group.add_argument("--fill", type=int, default=-1,
                            help="Launch serial program in the background so that this many threads are always running. If given, this will also become the upper limit for the number of threads instead of the number of cores of the machine. (default: -1 to disable")
    fill_group.add_argument("--bkgNice", type=int, default=None,
                            help="If given, use this 'nice' level for the background program")
    fill_group.add_argument("--bkgThreads", type=int, default=-1,
                            help="If given, use this many threads/process for the background program(s). (default: -1 for one process with necessary number of threads)")
    fill_group.add_argument("--cudaDevices", type=str, default="",
                            help="Comma-separeted list of CUDA devices (as in nvidia-smi) to use (default empty is to use all devices).")

    parser.add_argument("args", nargs=argparse.REMAINDER)

    opts = parseCommonArguments(parser)
    if opts.minThreads <= 0:
        parser.error("minThreads must be > 0, got %d" % opts.minThreads)
    if opts.maxThreads <= 0 and opts.maxThreads != -1:
        parser.error("maxThreads must be > 0 or -1, got %d" % opts.maxThreads)
    if opts.numThreads != "":
        opts.numThreads = [int(x) for x in opts.numThreads.split(",")]
    if opts.numStreams != "":
        opts.numStreams = [int(x) for x in opts.numStreams.split(",")]
    if opts.tasksetCores != "":
        opts.tasksetCores = opts.tasksetCores.split(",")
    if len(opts.tasksetCores) > 0 and opts.fill != -1 and len(opts.tasksetCores) != opts.fill:
        parser.error("When both --tasksetCores and --fill are given, --fill must match to the number of elements in --tasksetCores. No got --fill {} and {} elements in --tasksetCores".format(opts.fill, len(opts.tasksetCores)))
    if opts.runForMinutes >= 0:
        if opts.eventsPerStream is not None:
            parser.error("--runForMinutes and --eventsPerStream can not be used together")
        if opts.maxStreamsToAddEvents >= 0:
            parser.error("--runForMinutes and --maxStreamsToAddEvents can not be used together")
    opts.cudaDevices = opts.cudaDevices.split(",") if opts.cudaDevices != "" else []

    main(opts)
