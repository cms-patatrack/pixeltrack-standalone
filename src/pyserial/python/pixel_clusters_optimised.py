"""The clusterizer with the labelling reshaped rather than translated.

This module is pixel_clusters.py -- same decoding, same calibration, same
module boundaries, same charge cut, same numbering, same products, all of it in
pixel_clusters_common.py -- with one thing done differently: how the pixels of
a module are grouped into clusters.

findClus makes the pixels its nodes, propagates the minimum over every
neighbour pair every round, and stops when a round changes nothing.  Two things
about that are worth changing, and neither changes the answer:

  - a run of touching pixels within one column is one cluster by construction,
    so the labelling can take runs as its nodes.  There are only about a
    quarter fewer of them, but what they remove is exactly the chains the
    minimum would otherwise walk one edge at a time, and the rounds drop from
    27 an event to 12.
  - those rounds are wildly uneven: three of them settle all but a few dozen
    runs of forty thousand, and a stubborn cluster drags the rest out to
    nineteen, each still reducing over every edge.  A run's label can change
    only if a neighbour's did, so each round hands the next one the runs next
    to those that moved.

Together they are worth 4.1x on the labelling and 2.5x on the module; D27 has
the measurements, and pixel_clusters.py is what they are measured against.

Put it in place of pixel_clusters in any configuration, or use the
reco-optimised*.ini that already do:

    [clusters]
    @type = PythonProducer
    script = pixel_clusters_optimised
"""

import numpy as np

import edm_core  # built into the executable

import pixel_clusters_common as common


class PixelClustersOptimised:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.raw = registry.consumes(common.RAW, config.get("input", ""))
        self.digis = registry.produces(common.DIGIS)
        self.clusters = registry.produces(common.CLUSTERS)
        self.conditions = None

    @staticmethod
    def _runs(key):
        """Runs of touching pixels within one column, in the sorted order.

        A run is a maximal set of pixels of one column whose rows step by one,
        so every pixel of a run is in the same cluster by construction.  Runs
        rather than pixels are what the labelling works on.  There are only
        about a quarter fewer of them -- 1.38 pixels to a run on this data,
        since a cluster is usually longer in the column direction than across
        it -- but what they remove is exactly the chains the labelling is slow
        on: a run is a line of pixels the minimum would otherwise have to walk
        one edge at a time.
        """
        col = key // common.ROW_STRIDE
        row = key % common.ROW_STRIDE
        new = np.empty(key.size, dtype=bool)
        new[0] = True
        new[1:] = (col[1:] != col[:-1]) | (row[1:] - row[:-1] > 1)
        return np.cumsum(new) - 1, np.flatnonzero(new), col, row

    @staticmethod
    def _run_edges(runOf, key, col, row):
        """Runs of adjacent columns that touch.

        For each pixel, the pixels of the next column with a row within one are
        a contiguous stretch of the sorted order, which searchsorted finds at
        both ends at once; the runs those two pixels belong to are connected.
        """
        nextCol = (col + 1) * common.ROW_STRIDE
        lo = np.searchsorted(key, nextCol + row - 1, side="left")
        hi = np.searchsorted(key, nextCol + row + 1, side="right")
        counts = np.maximum(hi - lo, 0)
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        left = runOf[np.repeat(np.arange(runOf.size), counts)]
        right = runOf[np.repeat(lo, counts) + within]
        return left, right

    @staticmethod
    def _gather(source, starts, lengths, nodes):
        """The edges of `nodes`, laid end to end, and where each node's start.

        reduceat takes the ranges between the offsets it is given, so reducing
        over a subset of the nodes means copying their edges together first.
        """
        total = int(lengths.sum())
        offsets = np.cumsum(lengths) - lengths
        within = np.arange(total) - np.repeat(offsets, lengths)
        return source[np.repeat(starts[nodes], lengths) + within], offsets

    @classmethod
    def _label(cls, left, right, nRuns):
        """Which component each run belongs to, as the smallest run index in it.

        The C++ kernel iterates `min` over a neighbour list until nothing
        changes; this does the same over every module's runs at once, with two
        passes of pointer jumping per round so that a long cluster converges in
        the logarithm of its length rather than its length.

        The rounds are wildly uneven: three of them settle all but a few dozen
        runs of forty thousand, and then a handful of stubborn clusters take
        another fifteen.  So a round hands the next one only the runs next to
        those that moved -- a label can change only if a neighbour's did --
        and the tail stops costing a reduction over the whole edge list each
        time.  Laying out a subset is itself work, so a round whose front is
        still wider than a quarter of the runs reduces over everything
        instead.
        """
        label = np.arange(nRuns, dtype=np.int64)
        if left.size == 0:
            return label

        target = np.concatenate((left, right))
        source = np.concatenate((right, left))
        order = np.argsort(target, kind="stable")
        target = target[order]
        source = source[order]

        # every run's edges as one range, which is what lets a round reduce
        # over an arbitrary subset of the runs
        degree = np.bincount(target, minlength=nRuns)
        starts = np.cumsum(degree) - degree
        haveEdges = np.flatnonzero(degree)

        nodes = None  # None means every run that has an edge
        while True:
            before = label
            label = label.copy()
            if nodes is None:
                label[haveEdges] = np.minimum(label[haveEdges],
                                              np.minimum.reduceat(label[source], starts[haveEdges]))
            else:
                edges, offsets = cls._gather(source, starts, degree[nodes], nodes)
                label[nodes] = np.minimum(label[nodes], np.minimum.reduceat(label[edges], offsets))
            label = label[label]
            label = label[label]

            moved = np.flatnonzero(label != before)
            if moved.size == 0:
                return label
            if moved.size * 4 > nRuns:
                # still a broad front: laying out a subset costs more than
                # reducing over the whole edge list once
                nodes = None
                continue

            # the runs to look at next: those next to a run that moved, and
            # those that moved, in case something can still pull them lower.
            # A mask rather than unique(): sorting the front would cost more
            # than the round it saves.
            frontier = np.zeros(nRuns, dtype=bool)
            frontier[moved] = True
            neighbours, _ = cls._gather(source, starts, degree[moved], moved)
            frontier[neighbours] = True
            nodes = np.flatnonzero(frontier)
            nodes = nodes[degree[nodes] > 0]

    def _components(self, valid, moduleOf, xx, yy, nWords):
        """The label of every valid pixel: the lowest pixel index in its cluster.

        A different way of arriving at findClus's labelling, and the only
        thing this module does not share with pixel_clusters.py.
        """
        order, key = common.sorted_pixels(valid, moduleOf, xx, yy)

        runOf, runStart, col, row = self._runs(key)
        component = self._label(*self._run_edges(runOf, key, col, row), runStart.size)

        # which component each pixel belongs to, back in index order
        componentOf = np.empty(nWords, dtype=np.int64)
        componentOf[order] = component[runOf]
        pixelComponent = componentOf[valid]

        # a cluster is labelled by its lowest pixel: assigning the pixels to
        # their component in reverse leaves the first -- and therefore lowest
        # -- of each, since the later writes of a repeated index win
        root = np.empty(runStart.size, dtype=np.int64)
        root[pixelComponent[::-1]] = valid[::-1]
        return root[pixelComponent]

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        common.produce(self, event, eventSetup, self._components)


def create(config: dict, registry: edm_core.ProductRegistry) -> PixelClustersOptimised:
    return PixelClustersOptimised(config, registry)
