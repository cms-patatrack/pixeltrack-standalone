"""The histogram validation, in Python.

A port of HistoValidator: it fills thirty-three histograms of the digis,
clusters, rec hits, tracks and vertices of every event and writes them out at
the end of the job.  Like the C++ one it keeps no per-event state and produces
no product -- what it has instead is an accumulator shared by the whole job,
which here is a dict of numpy arrays under a lock rather than a map of atomic
counters.

The binning is SimpleAtomicHisto's: `nbins + 2` bins, with the first for
underflow and the last for overflow, and a value exactly at the top edge folded
back into the last real bin.  Both the arithmetic and the output format are
reproduced exactly, so the file this writes and the one the C++ module writes
are byte for byte the same.
"""

import threading

from namespaces import np

import edm_core  # built into the executable

DIGIS = "SiPixelDigisSoA"
CLUSTERS = "SiPixelClustersSoA"
REC_HITS = "TrackingRecHit2DCPU"
TRACKS = "PixelTrackHeterogeneous"
VERTICES = "ZVertexHeterogeneous"

QUALITY_LOOSE = 2

# name: (bins, min, max), in the order SimpleAtomicHisto's map holds them --
# which is sorted, and is the order they are written in
HISTOGRAMS = {
    "cluster_n": (200, 5000, 25000),
    "cluster_per_module_n": (110, 0, 110),
    "digi_adc": (250, 0, 5e4),
    "digi_n": (100, 0, 1e5),
    "hit_charge": (400, 0, 4e6),
    "hit_gr": (200, 0, 20),
    "hit_gx": (200, -20, 20),
    "hit_gy": (200, -20, 20),
    "hit_gz": (600, -60, 60),
    "hit_lex": (100, 0, 5e-5),
    "hit_ley": (100, 0, 1e-4),
    "hit_lx": (200, -1, 1),
    "hit_ly": (800, -4, 4),
    "hit_n": (200, 5000, 25000),
    "hit_sizex": (800, 0, 800),
    "hit_sizey": (800, 0, 800),
    "module_n": (100, 1500, 2000),
    "track_chi2": (100, 0, 40),
    "track_eta": (100, -3, 3),
    "track_n": (150, 0, 15000),
    "track_nhits": (3, 3, 6),
    "track_phi": (100, -3.15, 3.15),
    "track_pt": (400, 0, 400),
    "track_quality": (6, 0, 6),
    "track_tip": (100, -1, 1),
    "track_tip_zoom": (100, -0.05, 0.05),
    "track_zip": (100, -15, 15),
    "track_zip_zoom": (100, -0.1, 0.1),
    "vertex_chi2": (100, 0, 40),
    "vertex_n": (60, 0, 60),
    "vertex_ndof": (170, 0, 170),
    "vertex_pt2": (100, 0, 4000),
    "vertex_z": (100, -15, 15),
}

# The job's histograms, shared by every stream's instance of this module.
_lock = threading.Lock()
_counts = {name: np.zeros(bins + 2, dtype=np.int64) for name, (bins, _, _) in HISTOGRAMS.items()}


def _bins(name, values):
    """Which bin each value falls in, by SimpleAtomicHisto's rule."""
    nbins, low, high = HISTOGRAMS[name]
    low = np.float32(low)
    high = np.float32(high)
    v = np.asarray(values, dtype=np.float32).ravel()

    index = np.empty(v.size, dtype=np.int64)
    under = v < low
    over = v >= high
    inside = ~(under | over)
    scaled = ((v[inside] - low) / (high - low) * np.float32(nbins)).astype(np.int32)
    # a value just below the top edge can scale to nbins; the C++ folds it back
    scaled = np.where(scaled == nbins, nbins - 1, scaled)
    index[under] = 0
    index[over] = nbins + 1
    index[inside] = scaled + 1
    return np.bincount(index, minlength=nbins + 2)


class HistoValidator:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.digis = registry.consumes(DIGIS, config.get("input", ""))
        self.clusters = registry.consumes(CLUSTERS, config.get("input", ""))
        self.hits = registry.consumes(REC_HITS, config.get("hits", ""))
        self.tracks = registry.consumes(TRACKS, config.get("tracks", ""))
        self.vertices = registry.consumes(VERTICES, config.get("vertices", ""))
        self.output = config.get("output", "histograms_serial.txt")

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        filled = {}

        digis = event.get(self.digis)
        clusters = event.get(self.clusters)
        nDigis = digis.nDigis()
        nModules = digis.nModules()
        filled["digi_n"] = _bins("digi_n", [nDigis])
        filled["digi_adc"] = _bins("digi_adc", np.asarray(digis.adcSpan))
        filled["module_n"] = _bins("module_n", [nModules])
        filled["cluster_n"] = _bins("cluster_n", [clusters.nClusters()])
        filled["cluster_per_module_n"] = _bins("cluster_per_module_n",
                                               np.asarray(clusters.clusInModuleSpan)[:nModules])

        hits = event.get(self.hits).view()
        nHits = hits.nHits()
        filled["hit_n"] = _bins("hit_n", [nHits])
        for name, span in (("hit_lx", hits.xLocalSpan), ("hit_ly", hits.yLocalSpan),
                           ("hit_lex", hits.xerrLocalSpan), ("hit_ley", hits.yerrLocalSpan),
                           ("hit_gx", hits.xGlobalSpan), ("hit_gy", hits.yGlobalSpan),
                           ("hit_gz", hits.zGlobalSpan), ("hit_gr", hits.rGlobalSpan),
                           ("hit_charge", hits.chargeSpan),
                           ("hit_sizex", hits.clusterSizeXSpan), ("hit_sizey", hits.clusterSizeYSpan)):
            filled[name] = _bins(name, np.asarray(span)[:nHits])

        tracks = event.get(self.tracks).get()
        stride = tracks.stride()
        hitsPerTrack = np.diff(np.asarray(tracks.detIndices.off)[: stride + 1].astype(np.int64))
        quality = np.asarray(tracks.m_quality.data_)[:stride]
        good = (hitsPerTrack > 0) & (quality >= QUALITY_LOOSE)
        state = np.asarray(tracks.stateAtBS.state.data_)
        filled["track_n"] = _bins("track_n", [int(np.count_nonzero(good))])
        filled["track_nhits"] = _bins("track_nhits", hitsPerTrack[good])
        filled["track_chi2"] = _bins("track_chi2", np.asarray(tracks.chi2.data_)[:stride][good])
        filled["track_pt"] = _bins("track_pt", np.asarray(tracks.pt.data_)[:stride][good])
        filled["track_eta"] = _bins("track_eta", np.asarray(tracks.eta.data_)[:stride][good])
        filled["track_phi"] = _bins("track_phi", state[0:stride][good])
        filled["track_tip"] = _bins("track_tip", state[stride : 2 * stride][good])
        filled["track_tip_zoom"] = _bins("track_tip_zoom", state[stride : 2 * stride][good])
        filled["track_zip"] = _bins("track_zip", state[4 * stride : 5 * stride][good])
        filled["track_zip_zoom"] = _bins("track_zip_zoom", state[4 * stride : 5 * stride][good])
        filled["track_quality"] = _bins("track_quality", quality[good])

        vertices = event.get(self.vertices).get()
        nv = int(vertices.nvFinal)
        filled["vertex_n"] = _bins("vertex_n", [nv])
        filled["vertex_z"] = _bins("vertex_z", np.asarray(vertices.zv)[:nv])
        filled["vertex_chi2"] = _bins("vertex_chi2", np.asarray(vertices.chi2)[:nv])
        filled["vertex_ndof"] = _bins("vertex_ndof", np.asarray(vertices.ndof)[:nv])
        filled["vertex_pt2"] = _bins("vertex_pt2", np.asarray(vertices.ptv2)[:nv])

        with _lock:
            for name, counts in filled.items():
                _counts[name] += counts

    def endJob(self) -> None:
        with _lock:
            lines = []
            for name, (bins, low, high) in HISTOGRAMS.items():
                counts = " ".join(str(int(n)) for n in _counts[name])
                # "%g" is what an ostream prints a float as by default, which is
                # what the C++ module's dump() uses
                lines.append(f"{name} {bins + 2} {low:g} {high:g} {counts}\n")
        with open(self.output, "w") as out:
            out.writelines(lines)


def create(config: dict, registry: edm_core.ProductRegistry) -> HistoValidator:
    return HistoValidator(config, registry)
