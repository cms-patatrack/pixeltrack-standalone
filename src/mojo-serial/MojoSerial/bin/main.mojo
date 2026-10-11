from std.sys import argv
from std.ffi import external_call
from std.sys.terminate import exit
from std.time import perf_counter_ns
from std.pathlib import Path

from MojoSerial.bin.EventProcessor import EventProcessor
from MojoSerial.bin.PosixClockGettime import (
    PosixClockGettime,
    CLOCK_THREAD_CPUTIME_ID,
)
from MojoSerial.MojoBridge.DTypes import Double
from std.runtime.asyncrt import TaskGroup


def print_help(name: StringSlice):
    print(
        "Usage:",
        name,
        "[--warmupEvents WE] [--maxEvents ME] [--runForMinutes RM]",
        "[--threads N] [--data PATH] [--validation] [--histogram] [--empty]",
    )
    print(
        r"""(
Options:
  --warmupEvents                Number of events to process before starting the benchmark (default 0).
  --maxEvents                   Number of events to process (default -1 for all events in the input file).
  --runForMinutes               Continue processing the set of 1000 events until this many minutes have passed
                                (default -1 for disabled; conflicts with --maxEvents).
  --data                        Path to the 'data' directory (default 'data' in the directory of the executable).
  --validation                  Run (rudimentary) validation at the end.
  --empty                       Ignore all producers (for testing only).
  --threads                     Number of streams/threads to run concurrently (default 1).
)"""
    )


@always_inline
def parallelism_level() -> Int:
    """Gets the parallelism level of the Runtime.

    Returns:
        The number of worker threads available in the async runtime.
    """
    return Int(
        external_call[
            "KGEN_CompilerRT_AsyncRT_ParallelismLevel",
            Int32,
        ]()
    )

def main() raises:
    var args = argv()
    var warmupEvents = 0
    var maxEvents = -1
    var runForMinutes = -1
    var threads = 1
    var path = Path("")
    var empty = False
    var validation = False

    var i = 1
    while i < args.__len__():
        if args[i] == "-h" or args[i] == "--help":
            print_help(args[0])
            exit(0)
        elif args[i] == "--warmupEvents":
            i += 1
            warmupEvents = Int(args[i])
        elif args[i] == "--maxEvents":
            i += 1
            maxEvents = Int(args[i])
        elif args[i] == "--runForMinutes":
            i += 1
            runForMinutes = Int(args[i])
        elif args[i] == "--threads":
            i += 1
            threads = Int(args[i])
            print("Mojo is capping the parallelism level at", parallelism_level())
        elif args[i] == "--data":
            i += 1
            path = Path(args[i])
        elif args[i] == "--validation":
            validation = True
        elif args[i] == "--empty":
            empty = True
        else:
            print("Invalid parameter", args[i])
            print()
            exit(1)
        i += 1

    if maxEvents >= 0 and runForMinutes >= 0:
        print(
            "Got both --maxEvents and --runForMinutes, please give only one of"
            " them"
        )
        exit(1)
    if threads <= 0:
        print("Invalid --threads value, must be >= 1")
        exit(1)

    if not path:
        path = Path("data")

    if not path.exists():
        print("Data directory '", path, "' does not exist", sep="")
        exit(1)

    if runForMinutes < 0:
        print("Processing", maxEvents, "events", end="")
    else:
        print("Processing for about", runForMinutes, "minutes", end="")
    if warmupEvents > 0:
        print(" after", warmupEvents, "events of warm up", end="")
    print(", with ", threads, " concurrent events and ", threads, " threads.", sep="")


    var startEvent = List[Int](length=threads, fill=0)
    var endEvent = List[Int](length=threads, fill=0)

    if maxEvents > 0:
        var base = maxEvents // threads
        var rem = maxEvents % threads

        var offset = 0
        for t in range(threads):
            var count = base + (1 if t < rem else 0)
            startEvent[t] = offset
            endEvent[t] = offset + count
            offset += count
    else:
        for t in range(threads):
            startEvent[t] = 0
            endEvent[t] = -1

    var start = List[Int](length=threads, fill=0)
    var end = List[Int](length=threads, fill=0)

    var processed = List[Int](length=threads, fill=0)

    var cpu_start = List[Int](length=threads, fill=0)
    var cpu_end = List[Int](length=threads, fill=0)

    var processing_error = False

    # The event range comes in as arguments: read through this closure's
    # capture, startEvent[i]/endEvent[i] came back as heap addresses, so the
    # Source never reached its end event.
    async def worker(i: Int, stream_start: Int, stream_end: Int) capturing:
        var processor = EventProcessor(
            warmupEvents,
            stream_start,
            stream_end,
            runForMinutes,
            path,
            validation,
        )

        processor.warmUp()
        start[i] = perf_counter_ns()
        cpu_start[i] = PosixClockGettime[CLOCK_THREAD_CPUTIME_ID].now()
        processor.runToCompletion()
        end[i] = perf_counter_ns()
        cpu_end[i] = PosixClockGettime[CLOCK_THREAD_CPUTIME_ID].now()
        try:
            processor.endJob()
        except e:
            processing_error = True
            print("Error occurred while ending job ", i, ":", e)

        processed[i] = Int(processor.processedEvents())

    # `parallelize` is gone in 1.0; TaskGroup is the replacement. One task per
    # stream, scheduled across the async runtime's own pool.
    var tg = TaskGroup()
    for i in range(threads):
        tg.create_task(worker(i, startEvent[i], endEvent[i]))
    tg.wait()

    var diff = end[0] - start[0]
    for i in range(threads):
        diff = max(diff, end[i] - start[i])

    # in seconds
    var time: Double = Float64(diff) / 1e9

    var cpu_begin = cpu_start[0]/threads
    var cpu_stop = cpu_end[0]/threads
    for i in range(1, threads):
        cpu_begin = cpu_begin + cpu_start[i]/threads
        cpu_stop = cpu_stop + cpu_end[i]/threads

    var cpu_diff = cpu_stop - cpu_begin
    var cpu: Double = Float64(cpu_diff) / 1e9

    var totalEvents = 0
    for i in range(threads):
        totalEvents += processed[i]

    print(
        "Processed ",
        totalEvents,
        " events in ",
        time,
        " seconds, throughput ",
        (Float64(totalEvents) / time),
        " events/s, CPU usage per thread: ",
        round(cpu / time * 100),
        "%",
        sep="",
    )

    if processing_error:
        print("Processing completed with errors.")
        exit(1)
