"""The count validation, in Python.

A port of CountValidator: it reads the digi, cluster, track and vertex counts
the source carries alongside the raw data and checks what the chain produced
against them.  Tracks and vertices are compared within a tolerance, because
their counts depend on floating-point cuts that the reference was produced
with; digis, clusters and modules have to match exactly.

Two things about it are different in kind from the reconstruction modules.  It
has no product to publish -- it reads and reports -- and its counters belong to
the job rather than to a stream, so they live at module level, where every
stream's instance sees the same ones.  A free-threaded interpreter runs those
instances at the same time, so the updates are taken under a lock; endJob is
called on the first stream's module only, exactly as it is for a C++ module,
and prints what they add up to.
"""

import threading

import numpy as np

import edm_core  # built into the executable

DIGI_CLUSTER_COUNT = "DigiClusterCount"
TRACK_COUNT = "TrackCount"
VERTEX_COUNT = "VertexCount"
DIGIS = "SiPixelDigisSoA"
CLUSTERS = "SiPixelClustersSoA"
TRACKS = "PixelTrackHeterogeneous"
VERTICES = "ZVertexHeterogeneous"

TRACK_TOLERANCE = 0.012  # in 200 runs of 1k events all events are within this
VERTEX_TOLERANCE = 1

# The job's counters, shared by every stream's instance of this module.
_lock = threading.Lock()
_allEvents = 0
_goodEvents = 0
_trackDifference = 0.0
_vertexDifference = 0


class CountValidator:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.digiClusterCount = registry.consumes(DIGI_CLUSTER_COUNT, config.get("counts", ""))
        self.trackCount = registry.consumes(TRACK_COUNT, config.get("counts", ""))
        self.vertexCount = registry.consumes(VERTEX_COUNT, config.get("counts", ""))
        self.digis = registry.consumes(DIGIS, config.get("input", ""))
        self.clusters = registry.consumes(CLUSTERS, config.get("input", ""))
        self.tracks = registry.consumes(TRACKS, config.get("tracks", ""))
        self.vertices = registry.consumes(VERTICES, config.get("vertices", ""))

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        global _allEvents, _goodEvents, _trackDifference, _vertexDifference
        messages = []

        count = event.get(self.digiClusterCount)
        digis = event.get(self.digis)
        clusters = event.get(self.clusters)
        if digis.nModules() != count.nModules():
            messages.append(f"\n N(modules) is {digis.nModules()} expected {count.nModules()}")
        if digis.nDigis() != count.nDigis():
            messages.append(f"\n N(digis) is {digis.nDigis()} expected {count.nDigis()}")
        if clusters.nClusters() != count.nClusters():
            messages.append(f"\n N(clusters) is {clusters.nClusters()} expected {count.nClusters()}")

        # a track exists where it has hits; the tail of the SoA has none
        tracks = event.get(self.tracks).get()
        stride = tracks.stride()
        hits = np.diff(np.asarray(tracks.detIndices.off)[: stride + 1].astype(np.int64))
        nTracks = int(np.count_nonzero(hits > 0))
        expected = event.get(self.trackCount).nTracks()
        relative = abs(float(nTracks - expected) / expected)
        if nTracks != expected:
            trackDifference = relative
        else:
            trackDifference = 0.0
        if relative >= TRACK_TOLERANCE:
            messages.append(f"\n N(tracks) is {nTracks} expected {expected}, relative difference "
                            f"{relative} is outside tolerance {TRACK_TOLERANCE}")

        vertices = event.get(self.vertices).get()
        expected = event.get(self.vertexCount).nVertices()
        difference = abs(int(vertices.nvFinal) - int(expected))
        if difference > VERTEX_TOLERANCE:
            messages.append(f"\n N(vertices) is {vertices.nvFinal} expected {expected}, difference "
                            f"{difference} is outside tolerance {VERTEX_TOLERANCE}")

        with _lock:
            _allEvents += 1
            _trackDifference += trackDifference
            _vertexDifference += difference
            if not messages:
                _goodEvents += 1
        if messages:
            print(f"Event {event.eventID} " + "".join(messages), flush=True)

    def endJob(self) -> None:
        with _lock:
            allEvents, goodEvents = _allEvents, _goodEvents
            trackDifference, vertexDifference = _trackDifference, _vertexDifference
        if allEvents != goodEvents:
            print(f"CountValidator: {allEvents - goodEvents} events failed validation (see details above)", flush=True)
            raise RuntimeError("CountValidator failed")
        print(f"CountValidator: all {allEvents} events passed validation", flush=True)
        if trackDifference != 0.0:
            print(f" Average relative track difference {trackDifference / allEvents} (all within tolerance)",
                  flush=True)
        if vertexDifference != 0:
            print(f" Average absolute vertex difference {vertexDifference / allEvents} (all within tolerance)",
                  flush=True)


def create(config: dict, registry: edm_core.ProductRegistry) -> CountValidator:
    return CountValidator(config, registry)
