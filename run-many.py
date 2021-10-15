#!/usr/bin/env python3

import os
import json
import time
import socket
import argparse
import importlib
import subprocess
import collections

scan = importlib.import_module("run-scan")

RunningProgram = collections.namedtuple("RunningProgram", ["program", "index", "handle"])

class Program:
    def __init__(self, description):
        s = description.split(":")
        self._program = s[0]
        self._options = ["threads","streams","numa","cores","cudaDevices"]
        valid = set(self._options)
        for o in s[1:]:
            (name, value) = o.split("=")
            if name not in valid:
                raise Exception("Unsupported option '{}'".format(name))
            setattr(self, "_"+name, value)
            valid.remove(name)
        for o in valid:
            setattr(self, "_"+o, None)
        if self._threads is None:
            self._threads = 1
        if self._streams is None:
            self._streams = self._threads
        self._cudaDevices = self._cudaDevices.split(",") if self._cudaDevices is not None else []

    def program(self):
        return self._program

    def programShort(self):
        return os.path.basename(self._program)

    def cudaDevices(self):
        return self._cudaDevices

    def makeCommandMessage(self, processUntil):
        command = [self._program] + processUntil + ["--numberOfThreads", str(self._threads), "--numberOfStreams", str(self._streams)]
        msg = "Program {} threads {} streams {}".format(self.programShort(), self._threads, self._streams)
        if self._numa is not None:
            command = ["numactl", "--cpunodebind={}".format(self._numa), "--membind={}".format(self._numa)] + command
            msg += " NUMA node {}".format(self._numa)
        if self._cores is not None:
            command = ["taskset", "-c", self._cores]
            msg += " cores {}".format(self._cores)
        if len(self._cudaDevices) > 0:
            msg += " CUDA devices {}".format(",".join(self._cudaDevices))
        return (command, msg)

    def makeEnv(self, logfile):
        if len(self._cudaDevices) == 0:
            return os.environ
        visibleDevices = ",".join(self._cudaDevices)
        logfile.write("export CUDA_DEVICE_ORDER=PCI_BUS_ID\n")
        logfile.write("export CUDA_VISIBLE_DEVICES="+visibleDevices+"\n")
        logfile.flush()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=visibleDevices)
        return env

    def addMetadata(self, d):
        d["program"] = self.programShort()
        d["threads"] = self._threads
        d["streams"] = self._streams
        if self._numa is not None:
            d["numa"] = self._numa
        if self._cores is not None:
            d["cores"] = self._cores
        if self._cudaDevices is not None:
            d["cudaDevices"] = self._cudaDevices
            
    def __str__(self):
        ret = self._program+": "
        for o in self._options:
            v = getattr(self, "_"+o)
            if v is not None:
                ret += o+"="+v+" "
        return ret

class Monitor:
    def __init__(self, opts, programs, cudaDevices=[]):
        self._intervalSeconds = opts.monitorSeconds
        self._monitorMemory = opts.monitorMemory
        self._monitorCuda = opts.monitorCuda
        self._allPrograms = programs

        self._timeStamp = []
        self._dataMemory = [[] for p in programs]
        self._dataCuda = {x: [] for x in cudaDevices}
        self._dataCudaProcs = [{x: [] for x in p.cudaDevices()} for p in programs]

    def setIntervalSeconds(self, interval):
        self._intervalSeconds = interval

    def intervalSeconds(self):
        return self._intervalSeconds

    def snapshot(self, programs=[]):
        if self._intervalSeconds is None:
            return
        self._timeStamp.append(time.strftime("%y-%m-%d %H:%M:%S"))

        if self._monitorMemory:
            update = [dict(rss=0)]*len(self._dataMemory)
            for rp in programs:
                update[rp.index]["rss"] = scan.processRss(rp.handle.pid)
            for i, u in enumerate(update):
                self._dataMemory[i].append(u)

        if self._monitorCuda:
            for dev in self._dataCuda.keys():
                self._dataCuda[dev].append(scan.cudaDeviceStatus(dev)._asdict())
            update = [{x: dict(proc_mem_use=0) for x in p.cudaDevices()} for p in self._allPrograms]
            for rp in programs:
                for dev in rp.program.cudaDevices():
                    update[rp.index][dev]["proc_mem_use"] = scan.cudaDeviceProcessMemory(dev, rp.handle.pid)
            for i, u in enumerate(update):
                for dev, val in u.items():
                    self._dataCudaProcs[i][dev].append(val)

    def toArrays(self):
        data = {}
        if self._intervalSeconds is not None:
            data["time"] = self._timeStamp
            if self._monitorMemory:
                data["host"] = self._dataMemory
            if self._monitorCuda:
                data["cuda"] = dict(
                    device = self._dataCuda,
                    processes = self._dataCudaProcs,
                )
        return data

def runMany(programs, processUntil, opts, logfilenamebase, monitor):
    logfiles = []
    for i in range(0, len(programs)):
        logfiles.append(open(logfilenamebase.format(i), "w"))

    running_programs = []
    for i, (prog, logfile) in enumerate(zip(programs, logfiles)):
        (command, msg) = prog.makeCommandMessage(processUntil)
        msg = str(i) + " "+ msg + " minutes {}".format(opts.runForMinutes)
        scan.printMessage(msg)
        logfile.write(" ".join(command))
        logfile.write("\n----\n")
        logfile.flush()
        if opts.dryRun:
            print(" ".join(command))
            continue

        env = prog.makeEnv(logfile)
        p = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
        running_programs.append(RunningProgram(prog, i, p))
    monitor.snapshot(running_programs)
    finished_programs = []
    def terminate_programs():
        for p in running_programs:
            try:
                p.handle.terminate()
            except OSError:
                pass
            p.handle.wait()

    while len(running_programs) > 0:
        try:
            running_programs[0].handle.wait(timeout=monitor.intervalSeconds())
            rp = running_programs[0]
            del running_programs[0]
            finished_programs.append(rp)
            scan.printMessage("Program {} finished".format(rp.index))
            if rp.handle.returncode != 0:
                print(" got return code {}, aborting test".format(rp.handle.returncode))
                msg += "Program {} {} got return code {}, see output in log file {}, terminating the remaining programs.\n".format(rp.index, rp.program.program(), rp.handle.returncode, logfilenamebase.format(rp.index))
                terminate_programs()
        except subprocess.TimeoutExpired:
            monitor.snapshot(running_programs)
        except KeyboardInterrupt:
            terminate_programs()
    monitor.snapshot(running_programs)
    msg = ""
    for i, p in enumerate(running_programs):
        if p.returncode != 0:
            msg += "Program {} {} got return code %d, see output in log file %s\n".format(i, programs[i].program(), logfilenamebase.format(i))
    if len(msg) > 0:
        raise Exception(msg)

    for l in logfiles:
        l.close()

    ret = []
    for i in range(len(programs)):
        with open(logfilenamebase.format(i)) as logfile:
            ret.append(scan.throughput(logfile))
    return ret

            

def main(opts):
    programs = []
    cudaDevicesInPrograms = set()
    for x in opts.programs.split(";"):
        num = 1
        if x[0] == "[": 
            s = x[1:].split("]")
            num = int(s[0])
            x = s[1]
        for i in range(0, num):
            p = Program(x)
            programs.append(p)
            cudaDevicesInPrograms.update(p.cudaDevices())
    cudaDevicesInPrograms = list(cudaDevicesInPrograms)

    cudaDevicesAvailable = scan.listCudaDevices()
    print("Found {} devices".format(len(cudaDevicesAvailable)))
    for i, d in cudaDevicesAvailable.items():
        print(" {} {} driver {}".format(i, d.name, d.driver_version))
    for d in cudaDevicesInPrograms:
        if d not in cudaDevicesAvailable:
            raise Exception("Some program asked device {} but there is no device with that id".format(d))

    data = dict(
        args=" ".join(opts.args),
        results=[]
    )
    outputJson = opts.output+".json"
    if os.path.exists(outputJson):
        if opts.append:
            with open(outputJson) as inp:
                data = json.load(inp)
        elif not opts.overwrite:
            return

    hostname = socket.gethostname()

    if opts.runForMinutes < 0:
        raise Exception("currently --runForMinutes is required")
    mins = opts.runForMinutes

    tryAgain = opts.tryAgain
    while tryAgain > 0:
        try:
            monitor = Monitor(opts, programs, cudaDevices=cudaDevicesInPrograms)
            measurements = runMany(programs, ["--runForMinutes", str(mins)], opts, opts.output+"_log_{}.txt", monitor=monitor)
            break
        except Exception as e:
            tryAgain -= 1
            if tryAgain == 0:
                raise
                print("Got exception (see below), trying again ({} times left)".format(tryAgain))
                print("--------------------")
                print(str(e))
                print("--------------------")

    d = dict(
        hostname=hostname,
        throughput=sum([m.throughput for m in measurements]),
        programs=[]
    )
    scan.printMessage("Total throughput {} events/s".format(d["throughput"]))
    for i, (p, m) in enumerate(zip(programs, measurements)):
        dp = dict(
            events=m.events,
            time=m.time,
            throughput=m.throughput
        )
        p.addMetadata(dp)
        d["programs"].append(dp)
    if monitor.intervalSeconds() is not None:
        d["monitor"]=monitor.toArrays()
    if len(cudaDevicesInPrograms) > 0:
        d["cudaDevices"] = {
            x: dict(name=cudaDevicesAvailable[x].name, driver_version=cudaDevicesAvailable[x].driver_version) for x in cudaDevicesInPrograms
        }
    data["results"].append(d)
    
    with open(outputJson, "w") as out:
        json.dump(data, out, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Run given test programs.

Note that this program does not honor CUDA_VISIBLE_DEVICES, use cudaDevices instead.

Measuring combined throughput of multiple programs
[M]<program>:threads=N:streams=N:numa=N:cores=<list>:cudaDevices=<list>;<program>:...

  <program>   (Path to) the program to run
  threads     Number host threads (default: 1)
  streams     Number of streams (concurrent events) (default: same as threads)
  numa        NUMA node, uses 'numactl' (default: not set)
  cores       List of CPU cores to pin, uses 'taskset' (default: not set)
  cudaDevices List of CUDA devices to use (default: not set)
  M copies
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("programs", type=str,
                        help="Declaration of many programs to run (for syntax see above).")

    scan.addCommonArguments(parser)
    opts = scan.parseCommonArguments(parser)

    main(opts)
