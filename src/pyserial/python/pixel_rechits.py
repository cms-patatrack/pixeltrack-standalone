"""Pixel local reconstruction -- clusters to rec hits -- in Python.

A port of plugin-SiPixelRecHits/gpuPixelRecHits.h: for each cluster, the
extent and charge distribution over its digis, then the CPE position and its
error, then the local-to-global transform corrected for the beam spot.

This is the module that moves real data.  An event here has 48000-67000 digis
and 12000-21000 clusters, three orders of magnitude more objects than the
vertex finder sees, and every column of every product crosses as a numpy view
over the C++ storage: five columns of digis in, four of clusters, thirteen of
rec hits out.  Nothing is copied, and no binding here was written by hand.

The shape of the computation is a segmented reduction: each digi belongs to a
cluster, and the cluster's row and column extremes and its charge are the min,
max and sum over its digis.  numpy does that with the unbuffered ufunc methods
-- np.minimum.at and friends -- on a global cluster index, so the per-digi
loops in the C++ become four vectorised passes and no Python loop runs over
digis or clusters at all.

The detector geometry is different in kind from the event data: it comes from
the EventSetup and does not change from event to event, so the per-module
parameters are gathered into numpy arrays once, on the first event a stream
sees, and indexed by module thereafter.  Reading them per event would cost
1856 modules times twenty attribute reads, which is the one thing the boundary
is genuinely slow at.
"""

import numpy as np

import edm_core  # built into the executable

DIGIS = "SiPixelDigisSoA"
CLUSTERS = "SiPixelClustersSoA"
BEAM_SPOT = "BeamSpotPOD"
REC_HITS = "TrackingRecHit2DCPU"
CPE = "PixelCPEFast"

# phase1PixelTopology
INVALID_MODULE = 9999
NUM_ROWS_IN_ROC = 80
LAST_ROW_IN_ROC = 79
NUM_COLS_IN_ROC = 52
LAST_COL_IN_ROC = 51
LAST_ROW_IN_MODULE = 159
LAST_COL_IN_MODULE = 415
X_OFFSET = -81
Y_OFFSET = -54 * 4

MAX_CLUSTER_SIZE = 1023

# gpuPixelRecHits uses the largest uint16 rather than the per-module maximum,
# because pixmx is not in the binary dumps.
PIX_MAX = np.uint16(0xFFFF)


def _local_x(px):
    """phase1PixelTopology::localX, vectorised."""
    return px + (px > LAST_ROW_IN_ROC).astype(np.int32) + (px > NUM_ROWS_IN_ROC).astype(np.int32)


def _local_y(py):
    """phase1PixelTopology::localY, vectorised."""
    roc = py // NUM_COLS_IN_ROC
    in_roc = py - NUM_COLS_IN_ROC * roc
    return py + 2 * roc + (in_roc > 0).astype(np.int32)


def _is_big_x(px):
    return (px == 79) | (px == 80)


def _is_big_y(py):
    in_roc = py - NUM_COLS_IN_ROC * (py // NUM_COLS_IN_ROC)
    return (in_roc == 0) | (in_roc == LAST_COL_IN_ROC)


def _unsafe_atan2s(y, x):
    """DataFormats/approx_atan2.h unsafe_atan2s<7>, vectorised.

    The hits store phi as a short, from a degree-7 polynomial rather than a
    real atan2, so the port has to be the same polynomial: anything closer to
    the truth would disagree with the C++.
    """
    quarter = np.int32(32768 // 4)
    three_quarters = np.int32(3 * 32768 // 4)
    ax = np.abs(x)
    ay = np.abs(y)
    r = ((ax - ay) / (ax + ay)).astype(np.float32)
    r = np.where(x < 0, -r, r).astype(np.float32)
    z = (r * r).astype(np.float32)
    p = (r * (np.float32(-10422.177734375)
              + z * (np.float32(3349.97412109375)
                     + z * (np.float32(-1525.589599609375) + z * np.float32(406.64190673828125))))).astype(np.float32)
    angle = np.where(x >= 0, quarter, three_quarters) + p.astype(np.int16).astype(np.int32)
    return np.where(y < 0, -angle, angle).astype(np.int16)


class PixelRecHits:
    def __init__(self, config: dict, registry: edm_core.ProductRegistry) -> None:
        # An empty label means "the only product of that type", so the inputs
        # have to be named only in a job that runs more than one producer of
        # them -- reco-compare.ini does; the others do not.
        source = config.get("input", "")
        self.digis = registry.consumes(DIGIS, source)
        self.clusters = registry.consumes(CLUSTERS, source)
        self.beamSpot = registry.consumes(BEAM_SPOT, config.get("beamSpot", ""))
        self.hits = registry.produces(REC_HITS)
        self.geometry = None

    # ------------------------------------------------------------ geometry
    def _cache_geometry(self, cpe: edm_core.ParamsOnGPU) -> dict:
        """Every per-module parameter the CPE needs, as arrays indexed by module.

        Read once per stream: these are conditions, not event data.
        """
        common = cpe.commonParams()
        modules = LAST_MODULE = 1856
        fields = ("shiftX", "shiftY", "chargeWidthX", "chargeWidthY", "x0", "y0", "z0")
        geometry = {name: np.empty(modules, dtype=np.float32) for name in fields}
        geometry["isBarrel"] = np.empty(modules, dtype=bool)
        geometry["sx"] = np.empty((modules, 3), dtype=np.float32)
        geometry["sy"] = np.empty((modules, 3), dtype=np.float32)
        for name in ("px", "py", "pz", "r11", "r12", "r13", "r21", "r22", "r23"):
            geometry[name] = np.empty(modules, dtype=np.float32)

        for module in range(modules):
            det = cpe.detParams(module)
            for name in fields:
                geometry[name][module] = getattr(det, name)
            geometry["isBarrel"][module] = det.isBarrel
            geometry["sx"][module] = np.asarray(det.sx)
            geometry["sy"][module] = np.asarray(det.sy)
            frame = det.frame
            rotation = frame.rotation()
            geometry["px"][module] = frame.x()
            geometry["py"][module] = frame.y()
            geometry["pz"][module] = frame.z()
            geometry["r11"][module] = rotation.xx()
            geometry["r12"][module] = rotation.xy()
            geometry["r13"][module] = rotation.xz()
            geometry["r21"][module] = rotation.yx()
            geometry["r22"][module] = rotation.yy()
            geometry["r23"][module] = rotation.yz()

        geometry["thePitchX"] = np.float32(common.thePitchX)
        geometry["thePitchY"] = np.float32(common.thePitchY)
        geometry["theThicknessB"] = np.float32(common.theThicknessB)
        geometry["theThicknessE"] = np.float32(common.theThicknessE)
        return geometry

    # ---------------------------------------------------------- correction
    @staticmethod
    def _correction(size_m1, q_f, q_l, upper_first, lower_last, lorentz_shift,
                    thickness, cot_angle, pitch, first_is_big, last_is_big):
        """pixelCPEforGPU::correction, vectorised."""
        w_inner = pitch * (lower_last - upper_first).astype(np.float32)
        w_pred = thickness * cot_angle - lorentz_shift
        w_eff = np.abs(w_pred) - w_inner

        # size 2 uses the predicted width, unless it is inconsistent; every
        # other size uses the average edge length
        simple = (size_m1 != 1) | (w_eff < 0.0) | (w_eff > pitch)
        sum_of_edge = 2.0 + first_is_big.astype(np.float32) + last_is_big.astype(np.float32)
        w_eff = np.where(simple, pitch * 0.5 * sum_of_edge, w_eff)

        q_diff = (q_l - q_f).astype(np.float32)
        q_sum = (q_l + q_f).astype(np.float32)
        q_sum = np.where(q_sum == 0, np.float32(1.0), q_sum)
        return np.where(size_m1 == 0, np.float32(0.0), 0.5 * (q_diff / q_sum) * w_eff).astype(np.float32)

    # ------------------------------------------------------------- produce
    def produce(self, event: edm_core.Event, eventSetup: edm_core.EventSetup) -> None:
        cpe = eventSetup.get(CPE).getCPUProduct()
        if self.geometry is None:
            self.geometry = self._cache_geometry(cpe)
        geometry = self.geometry

        digis = event.get(self.digis)
        clusters = event.get(self.clusters)
        beamSpot = event.get(self.beamSpot)

        nHits = clusters.nClusters()
        product = event.allocate(self.hits, nHits, cpe, clusters.clusModuleStartSpan)
        hits = product.view()

        if nHits == 0 or digis.nModules() == 0:
            product.buildIndex()
            return

        moduleInd = np.asarray(digis.moduleIndSpan)
        clus = np.asarray(digis.clusSpan)
        clusModuleStart = np.asarray(clusters.clusModuleStartSpan)

        # Which cluster each digi belongs to, as a global hit index.  The C++
        # walks modules in moduleStart order and offsets by clusModuleStart of
        # the module it is in; a digi carries its own module, so the same index
        # falls out without the loop.
        valid = (moduleInd != INVALID_MODULE) & (clus >= 0)
        module = moduleInd[valid].astype(np.int32)
        hit = clusModuleStart[module] + clus[valid].astype(np.int32)
        inside = hit < nHits
        hit = hit[inside]
        module = module[inside]

        x = np.asarray(digis.xxSpan)[valid][inside].astype(np.int32)
        y = np.asarray(digis.yySpan)[valid][inside].astype(np.int32)
        charge = np.minimum(np.asarray(digis.adcSpan)[valid][inside], PIX_MAX).astype(np.int32)

        # The extent and the charge of each cluster: a segmented reduction over
        # its digis, which is what the two per-digi loops in the C++ are.
        minRow = np.full(nHits, np.iinfo(np.int32).max, dtype=np.int32)
        maxRow = np.zeros(nHits, dtype=np.int32)
        minCol = np.full(nHits, np.iinfo(np.int32).max, dtype=np.int32)
        maxCol = np.zeros(nHits, dtype=np.int32)
        np.minimum.at(minRow, hit, x)
        np.maximum.at(maxRow, hit, x)
        np.minimum.at(minCol, hit, y)
        np.maximum.at(maxCol, hit, y)

        totalCharge = np.zeros(nHits, dtype=np.int32)
        np.add.at(totalCharge, hit, charge)

        # The charge on the first and last row and column of each cluster,
        # which the position correction needs.  Known only once the extremes
        # are, so it is a second pass.
        q_f_x = np.zeros(nHits, dtype=np.int32)
        q_l_x = np.zeros(nHits, dtype=np.int32)
        q_f_y = np.zeros(nHits, dtype=np.int32)
        q_l_y = np.zeros(nHits, dtype=np.int32)
        for target, coordinate, edge in ((q_f_x, x, minRow), (q_l_x, x, maxRow),
                                         (q_f_y, y, minCol), (q_l_y, y, maxCol)):
            on_edge = coordinate == edge[hit]
            np.add.at(target, hit[on_edge], charge[on_edge])

        # Which module each cluster is in; every digi of a cluster agrees.
        detIndex = np.zeros(nHits, dtype=np.int32)
        detIndex[hit] = module

        # ---- pixelCPEforGPU::position
        llx = minRow + 1
        lly = minCol + 1
        llxl = _local_x(llx)
        llyl = _local_y(lly)
        urxl = _local_x(maxRow)
        uryl = _local_y(maxCol)

        xsize = urxl + 2 - llxl
        ysize = uryl + 2 - llyl
        xsize = xsize + _is_big_x(minRow).astype(np.int32) + _is_big_x(maxRow).astype(np.int32)
        ysize = ysize + _is_big_y(minCol).astype(np.int32) + _is_big_y(maxCol).astype(np.int32)

        with np.errstate(invalid="ignore", divide="ignore"):
            unbalanceX = (8.0 * np.abs((q_f_x - q_l_x).astype(np.float32)) / (q_f_x + q_l_x).astype(np.float32))
            unbalanceY = (8.0 * np.abs((q_f_y - q_l_y).astype(np.float32)) / (q_f_y + q_l_y).astype(np.float32))
        unbalanceX = np.nan_to_num(unbalanceX, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
        unbalanceY = np.nan_to_num(unbalanceY, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
        xsize = np.minimum(8 * xsize - unbalanceX, MAX_CLUSTER_SIZE)
        ysize = np.minimum(8 * ysize - unbalanceY, MAX_CLUSTER_SIZE)
        xsize = np.where((minRow == 0) | (maxRow == LAST_ROW_IN_MODULE), -xsize, xsize)
        ysize = np.where((minCol == 0) | (maxCol == LAST_COL_IN_MODULE), -ysize, ysize)

        pitchX = geometry["thePitchX"]
        pitchY = geometry["thePitchY"]
        xPos = geometry["shiftX"][detIndex] + pitchX * (0.5 * (llxl + urxl).astype(np.float32) + X_OFFSET)
        yPos = geometry["shiftY"][detIndex] + pitchY * (0.5 * (llyl + uryl).astype(np.float32) + Y_OFFSET)

        gvz = -1.0 / geometry["z0"][detIndex]
        cotalpha = (xPos - geometry["x0"][detIndex]) * gvz
        cotbeta = (yPos - geometry["y0"][detIndex]) * gvz

        isBarrel = geometry["isBarrel"][detIndex]
        thickness = np.where(isBarrel, geometry["theThicknessB"], geometry["theThicknessE"]).astype(np.float32)

        xcorr = self._correction(maxRow - minRow, q_f_x, q_l_x, llxl, urxl,
                                 geometry["chargeWidthX"][detIndex], thickness, cotalpha, pitchX,
                                 _is_big_x(minRow), _is_big_x(maxRow))
        ycorr = self._correction(maxCol - minCol, q_f_y, q_l_y, llyl, uryl,
                                 geometry["chargeWidthY"][detIndex], thickness, cotbeta, pitchY,
                                 _is_big_y(minCol), _is_big_y(maxCol))

        xLocal = (xPos + xcorr).astype(np.float32)
        yLocal = (yPos + ycorr).astype(np.float32)

        # ---- pixelCPEforGPU::errorFromDB
        sx = maxRow - minRow
        sy = maxCol - minCol
        isEdgeX = (minRow == 0) | (maxRow == LAST_ROW_IN_MODULE)
        isEdgeY = (minCol == 0) | (maxCol == LAST_COL_IN_MODULE)
        ix = (sx == 0).astype(np.int32) + ((sx == 0) & _is_big_x(minRow)).astype(np.int32)
        iy = (sy == 0).astype(np.int32) + ((sy == 0) & _is_big_y(minCol)).astype(np.int32)
        xerr = np.where(isEdgeX, np.float32(0.0050), geometry["sx"][detIndex, ix])
        yerr = np.where(isEdgeY, np.float32(0.0085), geometry["sy"][detIndex, iy])

        # ---- SOAFrame::toGlobal, then the beam spot
        xGlobal = (geometry["r11"][detIndex] * xLocal + geometry["r21"][detIndex] * yLocal
                   + geometry["px"][detIndex] - beamSpot.x).astype(np.float32)
        yGlobal = (geometry["r12"][detIndex] * xLocal + geometry["r22"][detIndex] * yLocal
                   + geometry["py"][detIndex] - beamSpot.y).astype(np.float32)
        zGlobal = (geometry["r13"][detIndex] * xLocal + geometry["r23"][detIndex] * yLocal
                   + geometry["pz"][detIndex] - beamSpot.z).astype(np.float32)

        # ---- store, straight into the product's own memory
        np.asarray(hits.chargeSpan)[:] = totalCharge
        np.asarray(hits.detectorIndexSpan)[:] = detIndex
        np.asarray(hits.xLocalSpan)[:] = xLocal
        np.asarray(hits.yLocalSpan)[:] = yLocal
        np.asarray(hits.clusterSizeXSpan)[:] = xsize
        np.asarray(hits.clusterSizeYSpan)[:] = ysize
        np.asarray(hits.xerrLocalSpan)[:] = xerr * xerr
        np.asarray(hits.yerrLocalSpan)[:] = yerr * yerr
        np.asarray(hits.xGlobalSpan)[:] = xGlobal
        np.asarray(hits.yGlobalSpan)[:] = yGlobal
        np.asarray(hits.zGlobalSpan)[:] = zGlobal
        np.asarray(hits.rGlobalSpan)[:] = np.sqrt(xGlobal * xGlobal + yGlobal * yGlobal)
        np.asarray(hits.iphiSpan)[:] = _unsafe_atan2s(yGlobal, xGlobal)

        product.buildIndex()


def create(config: dict, registry: edm_core.ProductRegistry) -> PixelRecHits:
    return PixelRecHits(config, registry)
