"""Pixel vertex reconstruction, in Python.

A port of plugin-PixelVertexFinding: loadTracks, clusterTracksByDensity,
fitVertices, splitVertices, fitVertices again, sortByPt2 -- the same sequence
gpuVertexFinder::Producer::make() runs, with the same parameters.

Nothing is copied across the boundary, and no binding here was written by
hand: every attribute this module touches is generated from the ROOT
dictionary of the product it belongs to.  `tracks.pt.data_` is a numpy view of
the C++ array, and the vertex SoA is constructed inside the Event and filled
through views of *its* storage, so what this module writes is the product.

The SoA layout is followed as the C++ declares it: eigenSoA stores component k
of element i of a MatrixSoA at data_[k * S + i], so the track z at the beam
spot is component 4 of `stateAtBS.state` and its variance is component 14 of
`stateAtBS.covariance`.

On fidelity: "find closest above me" keeps a running `mdist` in the C++ and
overwrites its answer each time it finds something nearer, so which of two
equally distant candidates wins depends on the scan order.  The C++ scans the
three histogram bins around each track; this takes the first candidate in z
order.  Ties are broken differently, but a tie needs two tracks at exactly
equal float distance, and the assignment is then percolated to the same seed
anyway in all but pathological cases.  splitVertices is a two-means iteration
per vertex, and its per-vertex work is small, so it stays a loop.

Everything else -- the selection, the neighbour counting, the search for the
closest denser track, the percolation, the weighted fits, the pt2 sums -- is
vectorised, over all the track pairs within eps in z at once.
"""

from namespaces import np

import edm_core  # built into the executable

TRACKS = "PixelTrackHeterogeneous"
VERTICES = "ZVertexHeterogeneous"

# trackQuality::Quality
QUALITY_LOOSE = 2

# The parameters PixelVertexProducerCUDA hard-codes.
MIN_T = 2  # min number of neighbours to be "seed"
EPS = 0.07  # max absolute distance to cluster
ERRMAX = 0.01  # max error to be "seed"
CHI2MAX = 9.0  # max normalized distance to cluster
PT_MIN = 0.5  # GeV

INT8_MIN = -128


class VertexFinder:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        # An empty label means "the only product of that type"; see pixel_rechits.
        self.tracks = registry.consumes(TRACKS, config.get("input", ""))
        self.vertices = registry.produces(VERTICES)
        self.ptMin = float(config.get("ptMin", PT_MIN))
        self.minT = int(config.get("minT", MIN_T))
        self.eps = float(config.get("eps", EPS))
        self.errmax = float(config.get("errmax", ERRMAX))
        self.chi2max = float(config.get("chi2max", CHI2MAX))

    # ------------------------------------------------------------ loadTracks
    @staticmethod
    def _component(matrix, index, stride):
        """Component `index` of an eigenSoA MatrixSoA, as a contiguous view."""
        return matrix.data_[index * stride : (index + 1) * stride]

    def _select(self, tracks):
        """The tracks the vertex finder uses, and their z, error^2 and pt^2.

        Mirrors loadTracks(): at least four hits, loose quality, pt above the
        cut.  The C++ stops at the first track with no hits, which is the guard
        for the unfilled tail of the SoA; the same bound is found here with
        argmin on the hit counts.
        """
        stride = tracks.stride()
        nHits = np.diff(tracks.detIndices.off[: stride + 1].astype(np.int64))

        empty = np.flatnonzero(nHits == 0)
        limit = int(empty[0]) if empty.size else stride

        quality = tracks.m_quality.data_[:limit]
        pt = tracks.pt.data_[:limit]
        keep = (nHits[:limit] >= 4) & (quality == QUALITY_LOOSE) & (pt >= self.ptMin)

        zip_ = self._component(tracks.stateAtBS.state, 4, stride)
        zipError2 = self._component(tracks.stateAtBS.covariance, 14, stride)

        itrk = np.flatnonzero(keep).astype(np.uint16)
        zt = zip_[:limit][keep].astype(np.float32)
        ezt2 = zipError2[:limit][keep].astype(np.float32)
        ptt2 = (pt[keep] * pt[keep]).astype(np.float32)
        return itrk, zt, ezt2, ptt2, limit

    # ------------------------------------------- clusterTracksByDensity
    def _cluster(self, zt, ezt2):
        nt = zt.size
        er2mx = self.errmax * self.errmax

        iv = np.arange(nt, dtype=np.int32)

        # Candidate neighbours.  The C++ scans the three histogram bins around
        # each track and then rejects anything further than eps; since a bin is
        # 0.1 wide and eps is 0.07, that is exactly the tracks within eps in z,
        # which is what this window selects directly.
        #
        # The sort has to be on z, not on the bin index: within a bin the z
        # values are in no particular order, and a binary search over them
        # returns nonsense bounds.
        order = np.argsort(zt, kind="stable")
        zsorted = zt[order]
        lo = np.searchsorted(zsorted, zt - self.eps, side="left")
        hi = np.searchsorted(zsorted, zt + self.eps, side="right")

        # Every (track, candidate) pair in the windows, in window order: the
        # candidates of track i are order[lo[i]:hi[i]], so pair k belongs to
        # track pi[k] and its candidate is order[lo[pi[k]] + k - first[pi[k]]].
        counts = (hi - lo).astype(np.int64)
        first = np.cumsum(counts) - counts
        pi = np.repeat(np.arange(nt, dtype=np.int32), counts)
        pj = order[np.repeat(lo, counts) + np.arange(pi.size) - np.repeat(first, counts)]
        dist = np.abs(zt[pi] - zt[pj])
        close = (dist <= self.eps) & (dist * dist <= self.chi2max * (ezt2[pi] + ezt2[pj]))

        # The number of neighbours of each seed.
        seeds = ezt2 <= er2mx
        counted = close & (pj != pi) & seeds[pi]
        nn = np.bincount(pi[counted], minlength=nt).astype(np.int32)

        # "find closest above me": the nearest track that is denser, or equally
        # dense and lower in z.  Among equally near candidates the first one in
        # window order wins, as argmin over the window would choose.
        better = (nn[pj] > nn[pi]) | ((nn[pj] == nn[pi]) & (zt[pj] < zt[pi]))
        cand = np.flatnonzero(close & better)
        if cand.size:
            ci = pi[cand]
            cd = dist[cand]
            starts = np.flatnonzero(np.r_[True, ci[1:] != ci[:-1]])
            nearest = np.minimum.reduceat(cd, starts)
            at_min = cd == np.repeat(nearest, np.diff(np.r_[starts, ci.size]))
            winners, firsts = np.unique(ci[at_min], return_index=True)
            iv[winners] = pj[cand[at_min][firsts]]

        # Consolidate the graph: percolate to the seed of each cluster.  Each
        # track points at a denser one, or an equally dense one lower in z, so
        # there are no cycles and pointer jumping reaches the same seed as
        # following the chain track by track.
        while True:
            jumped = iv[iv]
            if np.array_equal(jumped, iv):
                break
            iv = jumped

        # A track that points at itself and is dense enough is a cluster; the
        # rest is noise.
        isSeed = iv == np.arange(nt, dtype=np.int32)
        dense = isSeed & (nn >= self.minT)
        noise = isSeed & ~dense

        clusterId = np.full(nt, -1, dtype=np.int32)
        clusterId[dense] = np.arange(int(np.count_nonzero(dense)), dtype=np.int32)
        foundClusters = int(np.count_nonzero(dense))

        out = np.empty(nt, dtype=np.int32)
        out[:] = clusterId[iv]
        out[noise[iv]] = 9998 + 1  # the C++ marks noise as -9998, then negates
        return out, nn, foundClusters

    # ----------------------------------------------------------- fitVertices
    def _fit(self, iv, zt, ezt2, nvFinal, zv, wv, chi2, ndof, chi2Max):
        good = iv <= 9990
        if nvFinal == 0:
            return
        zv[:nvFinal] = 0.0
        wv[:nvFinal] = 0.0
        chi2[:nvFinal] = 0.0

        w = np.zeros_like(ezt2)
        np.divide(1.0, ezt2, out=w, where=good)

        np.add.at(zv[:nvFinal], iv[good], (zt * w)[good])
        np.add.at(wv[:nvFinal], iv[good], w[good])

        nonzero = wv[:nvFinal] > 0
        zv[:nvFinal][nonzero] /= wv[:nvFinal][nonzero]
        ndof[:nvFinal] = -1

        c2 = np.zeros_like(zt)
        c2[good] = zv[:nvFinal][iv[good]] - zt[good]
        c2[good] = c2[good] * c2[good] / ezt2[good]

        outlier = good & (c2 > chi2Max)
        iv[outlier] = 9999
        kept = good & ~outlier

        np.add.at(chi2[:nvFinal], iv[kept], c2[kept])
        np.add.at(ndof[:nvFinal], iv[kept], 1)

        split = ndof[:nvFinal] > 0
        wv[:nvFinal][split] *= ndof[:nvFinal][split] / chi2[:nvFinal][split]

    # --------------------------------------------------------- splitVertices
    def _split(self, iv, zt, ezt2, nvFinal, zv, wv, chi2, ndof, maxChi2):
        nvIntermediate = nvFinal
        for kv in range(nvFinal):
            if ndof[kv] < 4:
                continue
            if chi2[kv] < maxChi2 * float(ndof[kv]):
                continue

            members = np.flatnonzero(iv == kv)
            nq = members.size
            if nq >= 512:  # MAXTK in the C++
                continue

            zz = (zt[members] - zv[kv]).astype(np.float32)
            ww = (1.0 / ezt2[members]).astype(np.float32)
            newV = (zz >= 0).astype(np.uint8)

            znew = np.zeros(2, dtype=np.float32)
            wnew = np.zeros(2, dtype=np.float32)
            for _ in range(20):
                znew[:] = 0.0
                wnew[:] = 0.0
                np.add.at(znew, newV, zz * ww)
                np.add.at(wnew, newV, ww)
                if wnew[0] == 0 or wnew[1] == 0:
                    break
                znew /= wnew
                nearer = (np.abs(zz - znew[0]) >= np.abs(zz - znew[1])).astype(np.uint8)
                if np.array_equal(nearer, newV):
                    break
                newV = nearer

            if wnew[0] == 0 or wnew[1] == 0:
                continue

            dist2 = (znew[0] - znew[1]) ** 2
            if dist2 / (1.0 / wnew[0] + 1.0 / wnew[1]) < 4:
                continue

            iv[members[newV == 1]] = nvIntermediate
            nvIntermediate += 1
        return nvIntermediate

    # ------------------------------------------------------------- sortByPt2
    def _sort(self, iv, itrk, ptt2, nvFinal, idv, ptv2, sortInd):
        if nvFinal < 1:
            return
        idv[itrk.astype(np.int64)] = iv.astype(np.int16)
        ptv2[:nvFinal] = 0.0
        good = iv <= 9990
        np.add.at(ptv2[:nvFinal], iv[good], ptt2[good])
        sortInd[:nvFinal] = np.argsort(ptv2[:nvFinal], kind="stable").astype(np.uint16)

    # --------------------------------------------------------------- produce
    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        tracks = event.get(self.tracks).get()
        vertices = event.allocate(self.vertices).get()

        idv = vertices.idv
        zv = vertices.zv
        wv = vertices.wv
        chi2 = vertices.chi2
        ptv2 = vertices.ptv2
        ndof = vertices.ndof
        sortInd = vertices.sortInd

        itrk, zt, ezt2, ptt2, limit = self._select(tracks)
        idv[:limit] = -1
        if zt.size == 0:
            vertices.nvFinal = 0
            return

        iv, nn, nvFinal = self._cluster(zt, ezt2)
        ndof[: len(nn)] = nn

        self._fit(iv, zt, ezt2, nvFinal, zv, wv, chi2, ndof, 50.0)
        nvFinal = self._split(iv, zt, ezt2, nvFinal, zv, wv, chi2, ndof, 9.0)
        self._fit(iv, zt, ezt2, nvFinal, zv, wv, chi2, ndof, 5000.0)
        self._sort(iv, itrk, ptt2, nvFinal, idv, ptv2, sortInd)

        vertices.nvFinal = nvFinal


def create(config: dict, registry: edm_core.ProductRegistry) -> VertexFinder:
    return VertexFinder(config, registry)
