"""Raw data to digis to clusters, in Python.

A port of SiPixelRawToClusterCUDA and the kernels behind it: the FED walk in
the producer, SiPixelRawToClusterGPUKernel's RawToDigi, gpuCalibPixel's
calibDigis, gpuClustering's countModules, findClus and clusterChargeCut, and
the capped prefix sum that gives the rec hits their module offsets.

This is the first module of the chain and the only one whose input is not
already a column: it reads the FED buffers as bytes and decodes them.

The kernels are translated rather than rethought.  Where the C++ loops over
pixels this computes a column, which is what numpy is for and what every other
module in this backend does; the quantities, their order and the branches are
still the C++'s.  findClus in particular is the algorithm the kernel has: the
pixels themselves are the nodes, the neighbours of a pixel are the ones that
come after it in the column histogram and are within one row -- the rest of its
own column, and all of the next one -- and the minimum is propagated over every
pair, every round, until a round changes nothing, with the odd rounds following
each label to its root.

python/pixel_clusters_optimised.py is this module with a different labelling,
and D27 measures what that is worth.  Everything the two share is in
pixel_clusters_common.py.

The digi errors product is not written here.  The C++ module produces it and
nothing consumes it.
"""

import numpy as np

import edm_core  # built into the executable

import pixel_clusters_common as common


class PixelClusters:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        self.raw = registry.consumes(common.RAW, config.get("input", ""))
        self.digis = registry.produces(common.DIGIS)
        self.clusters = registry.produces(common.CLUSTERS)
        self.conditions = None

    @staticmethod
    def _neighbours(order, key, col, row):
        """The neighbour list findClus builds, for every module at once.

        From each pixel, everything later in the column histogram within one
        row of it: the rest of its own column, then all of the next one.  Both
        are stretches of the sorted order, so searchsorted finds each of them
        at both ends at once.

        The C++ walks its own column in fill order rather than in row order,
        which picks a different half of each pair; since the minimum is
        propagated both ways along an edge, the two give the same labelling.
        """
        position = np.arange(order.size)
        ranges = [
            # the rest of this column, up to one row away
            (position + 1, np.searchsorted(key, col * common.ROW_STRIDE + row + 1, side="right")),
            # all of the next column, within one row
            (np.searchsorted(key, (col + 1) * common.ROW_STRIDE + row - 1, side="left"),
             np.searchsorted(key, (col + 1) * common.ROW_STRIDE + row + 1, side="right")),
        ]

        left = []
        right = []
        for lo, hi in ranges:
            counts = np.maximum(hi - lo, 0)
            total = int(counts.sum())
            if total == 0:
                continue
            within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            left.append(np.repeat(order, counts))
            right.append(order[np.repeat(lo, counts) + within])
        if not left:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(left), np.concatenate(right)

    def _components(self, valid, moduleOf, xx, yy, nWords):
        """findClus: the `while more` loop, over every module's pixels at once.

        The label of a pixel starts as its own index, as countModules leaves
        it, and each round takes the minimum over the neighbour list and then
        follows every label to its root -- the two things the C++ alternates
        between on even and odd rounds.  It converges to the lowest pixel index
        in each cluster, which is what makes the numbering afterwards the same.
        """
        order, key = common.sorted_pixels(valid, moduleOf, xx, yy)

        label = np.arange(nWords, dtype=np.int64)
        left, right = self._neighbours(order, key, key // common.ROW_STRIDE, key % common.ROW_STRIDE)
        if left.size == 0:
            return label[valid]

        # an edge pulls both of its pixels, so it is held both ways round, and
        # grouping by the one being pulled turns the scatter into a reduction:
        # np.minimum.at cannot buffer and is an order of magnitude slower over
        # the same edges
        target = np.concatenate((left, right))
        source = np.concatenate((right, left))
        order_ = np.argsort(target, kind="stable")
        target = target[order_]
        source = source[order_]
        starts = np.flatnonzero(np.concatenate(([True], target[1:] != target[:-1])))
        group = target[starts]

        while True:
            before = label
            label = label.copy()
            label[group] = np.minimum(label[group], np.minimum.reduceat(label[source], starts))
            while True:  # the odd rounds: follow each label to its root
                jumped = label[label]
                if np.array_equal(jumped, label):
                    break
                label = jumped
            if np.array_equal(label, before):
                return label[valid]

    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        common.produce(self, event, eventSetup, self._components)


def create(config: dict, registry: edm_core.ProductRegistry) -> PixelClusters:
    return PixelClusters(config, registry)
