"""What the two clusterizer modules have in common.

pixel_clusters.py and pixel_clusters_optimised.py differ in one thing: how the
pixels of a module are grouped into clusters.  Everything else -- the FED walk,
the decoding, the calibration, the module boundaries, the charge cut, the
numbering and the products -- is the same work, and it lives here so that a fix
to it cannot reach one of them and not the other.

That split is the translation on one side and the algorithm on the other.  What
is here is a translation of the C++ kernel by kernel, with a column where the
C++ has a loop over pixels, which is what numpy is for; what is not here is the
labelling, the one place where the two modules make different choices.  D27
measures what each of the two is worth.
"""

import numpy as np

import edm_core  # built into the executable

RAW = "FEDRawDataCollection"
DIGIS = "SiPixelDigisSoA"
CLUSTERS = "SiPixelClustersSoA"
FED_IDS = "SiPixelFedIds"
CABLING = "SiPixelFedCablingMapGPUWrapper"
GAINS = "SiPixelGainCalibrationForHLTGPU"

# ------------------------------------------------------------------ constants
MAX_FED = 150
MAX_LINK = 48
MAX_ROC = 8
MAX_WORD = 2000
MAX_FED_WORDS = MAX_FED * MAX_WORD
FED_ID_OFFSET = 1200
PILOT_BLADE_FED = 40

MAX_NUM_MODULES = 2000
MAX_HITS_IN_MODULE = 1024
MAX_NUM_CLUSTERS = 48 * 1024
MAX_PIX_IN_MODULE = 4000
INV_ID = 9999

NUM_ROWS_IN_ROC = 80
NUM_COLS_IN_ROC = 52
MAX_ROC_INDEX = 8

# the raw word, laid out as in SiPixelRawToClusterGPUKernel.h
ADC_MASK = 0xFF
PXID_SHIFT, PXID_MASK = 8, 0xFF
DCOL_SHIFT, DCOL_MASK = 16, 0x1F
ROC_SHIFT, ROC_MASK = 21, 0x1F
LINK_SHIFT, LINK_MASK = 26, 0x3F
ROW_SHIFT, ROW_MASK = 8, 0x7F  # layer 1 only
COL_SHIFT, COL_MASK = 15, 0x3F  # layer 1 only
ERROR_MASK = 0x1F

LAYER_START_BIT, LAYER_MASK = 20, 0xF
MODULE_START_BIT, MODULE_MASK = 2, 0x3FF
PANEL_START_BIT, PANEL_MASK = 10, 0x3

# the error codes that mean "skip this ROC"
SKIP_ERRORS = (26, 31)  # the contiguous range 26..31; 25 is checked against the cabling

# pdigi packs 11 bits of row, 11 of column and 10 of charge
PACK_COLUMN_SHIFT = 11
PACK_ADC_SHIFT = 22
PACK_MAX_ADC = 1023

# calibration, valid for run 2
VCAL_TO_ELECTRON_GAIN = np.float32(47.0)
VCAL_TO_ELECTRON_GAIN_L1 = np.float32(50.0)
VCAL_TO_ELECTRON_OFFSET = np.float32(-60.0)
VCAL_TO_ELECTRON_OFFSET_L1 = np.float32(-670.0)
LAYER_1_MODULES = 96  # modules 0..95 are the barrel's first layer

CHARGE_CUT_L1 = 2000
CHARGE_CUT = 4000

# the sort key that puts a module's pixels in column-then-row order
COL_STRIDE = 512
ROW_STRIDE = 256


def cache_conditions(eventSetup: edm_core.EventSetup) -> dict:
    """The cabling map, the gain ranges and the FED list, as columns.

    Once per stream: none of it changes from event to event, and the gain
    ranges are 2000 modules' worth of scalars reached one bound call at a
    time -- a scalar attribute read each (D20), which is not a per-event cost
    worth paying.
    """
    cabling = eventSetup.get(CABLING)
    table = cabling.cablingMap()
    wrapper = eventSetup.get(GAINS)
    gain = wrapper.gains()
    modules = range(MAX_NUM_MODULES)
    return {
        "fedIds": np.asarray(eventSetup.get(FED_IDS).fedIdsSpan).copy(),
        "rawId": np.asarray(table.RawId).copy(),
        "rocInDet": np.asarray(table.rocInDet).copy(),
        "moduleId": np.asarray(table.moduleId).copy(),
        "badRocs": np.asarray(table.badRocs).copy(),
        "link": np.asarray(table.link).copy(),
        "roc": np.asarray(table.roc).copy(),
        "size": int(table.size),
        "modToUnp": np.asarray(cabling.modToUnpAll).copy(),
        "pedestals": np.asarray(wrapper.pedestals),
        "rangeFirst": np.array([gain.rangeFirst(m) for m in modules], dtype=np.int64),
        "rangeLast": np.array([gain.rangeLast(m) for m in modules], dtype=np.int64),
        "nCols": np.array([gain.numberOfCols(m) for m in modules], dtype=np.int64),
        "minPed": np.float32(gain.minPed_),
        "pedPrecision": np.float32(gain.pedPrecision),
        "minGain": np.float32(gain.minGain_),
        "gainPrecision": np.float32(gain.gainPrecision),
        "deadFlag": int(gain.deadFlag_),
        "noisyFlag": int(gain.noisyFlag_),
        "rowsAveragedOver": int(gain.numberOfRowsAveragedOver_),
    }


def fed_words(raw, fedIds):
    """The payload words of every FED, and which FED each pair came from.

    The walk the C++ producer does: skip the pilot blade, drop a FED whose
    trailer has the CRC bit set, then step over the headers from the front
    and the trailers from the back -- both chain through a "more" bit --
    and take what is left between them.
    """
    words = []
    feds = []
    for fedId in fedIds:
        if fedId == PILOT_BLADE_FED:
            continue
        data = np.asarray(raw.FEDData(int(fedId)).dataSpan)
        n64 = data.size // 8
        if n64 == 0:
            continue
        w64 = data[: n64 * 8].view(np.uint64)
        if (int(w64[-1]) >> 2) & 1:  # CRC error
            continue

        head = 0  # the last header word
        while head < n64:
            word = int(w64[head])
            if (word >> 60) & 0xF != 0x5:  # eventid's control id: not a header
                break
            if not (word >> 3) & 0x1:  # sourceid's "more headers" bit
                break
            head += 1

        tail = n64 - 1  # the first trailer word, scanning backwards
        while tail > 0:
            word = int(w64[tail])
            if (word >> 60) & 0xF != 0xA:  # eventsize's control id: not a trailer
                break
            if not (word >> 3) & 0x1:  # conscheck's "more trailers" bit
                break
            tail -= 1

        payload = data[(head + 1) * 8 : tail * 8].view(np.uint32)
        if payload.size == 0:
            continue
        words.append(payload)
        feds.append(np.full(payload.size // 2, fedId - FED_ID_OFFSET, dtype=np.int64))

    if not words:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.int64)
    return np.concatenate(words), np.concatenate(feds)


def raw_to_digi(words, feds, c):
    """Every word decoded at once: bit fields, the cabling lookup, and the
    ROC-local to module-local frame conversion."""
    n = words.size
    fed = np.repeat(feds, 2)[:n]

    link = ((words >> LINK_SHIFT) & LINK_MASK).astype(np.int64)
    roc = ((words >> ROC_SHIFT) & ROC_MASK).astype(np.int64)
    index = np.clip(fed * (MAX_LINK * MAX_ROC) + (link - 1) * MAX_ROC + roc, 0, c["rawId"].size - 1)

    rawId = c["rawId"][index].astype(np.int64)
    rocInDet = c["rocInDet"][index].astype(np.int64)
    moduleId = c["moduleId"][index].astype(np.int64)

    # checkROC: an error word of 25 or more that the cabling agrees with
    errorType = ((words >> ROC_SHIFT) & ERROR_MASK).astype(np.int64)
    found = (errorType >= SKIP_ERRORS[0]) & (errorType <= SKIP_ERRORS[-1])
    is25 = errorType == 25
    if is25.any():
        index25 = np.clip(fed * (MAX_LINK * MAX_ROC) + (link - 1) * MAX_ROC + 1, 0, c["link"].size - 1)
        inRange = (index25 > 1) & (index25 <= c["size"])
        agrees = (link == c["link"][index25]) & (c["roc"][index25] == 1)
        found |= is25 & (~inRange | agrees)
    skip = (roc >= MAX_ROC_INDEX) & found
    skip |= c["badRocs"][index] != 0  # quality
    skip |= c["modToUnp"][index] != 0  # modules not to unpack

    barrel = ((rawId >> 25) & 0x7) == 1
    layer = np.where(barrel, (rawId >> LAYER_START_BIT) & LAYER_MASK, 0)
    module = (rawId >> MODULE_START_BIT) & MODULE_MASK
    panel = (rawId >> PANEL_START_BIT) & PANEL_MASK
    side = np.where(barrel, np.where(module < 5, -1, 1), np.where(panel == 1, -1, 1))

    # layer 1 sends a row and a column; everything else a double column and
    # a pixel index within it
    isL1 = layer == 1
    col1 = ((words >> COL_SHIFT) & COL_MASK).astype(np.int64)
    row1 = ((words >> ROW_SHIFT) & ROW_MASK).astype(np.int64)
    dcol = ((words >> DCOL_SHIFT) & DCOL_MASK).astype(np.int64)
    pxid = ((words >> PXID_SHIFT) & PXID_MASK).astype(np.int64)
    row = np.where(isL1, row1, NUM_ROWS_IN_ROC - pxid // 2)
    col = np.where(isL1, col1, dcol * 2 + pxid % 2)
    valid = np.where(isL1,
                     (row1 < NUM_ROWS_IN_ROC) & (col1 < NUM_COLS_IN_ROC),
                     (dcol < 26) & (pxid >= 2) & (pxid < 162))

    # frameConversion.  Two orientations: which one a module has depends on
    # the side and the layer, and both endcap panels share the first.
    flipped = barrel & ((side == 1) | isL1)
    low = rocInDet < 8
    slopeRow = np.where(flipped, np.where(low, -1, 1), np.where(low, 1, -1))
    rowOffset = np.where(slopeRow < 0, 2 * NUM_ROWS_IN_ROC - 1, 0)
    colOffset = np.where(
        flipped,
        np.where(low, rocInDet * NUM_COLS_IN_ROC, (16 - rocInDet) * NUM_COLS_IN_ROC - 1),
        np.where(low, (8 - rocInDet) * NUM_COLS_IN_ROC - 1, (rocInDet - 8) * NUM_COLS_IN_ROC),
    )

    slopeCol = -slopeRow
    keep = (words != 0) & ~skip & valid
    xx = np.where(keep, rowOffset + slopeRow * row, 0).astype(np.uint16)
    yy = np.where(keep, colOffset + slopeCol * col, 0).astype(np.uint16)
    adc = np.where(keep, words & ADC_MASK, 0).astype(np.uint16)
    moduleInd = np.where(keep, moduleId, INV_ID).astype(np.uint16)
    rawIdArr = np.where(keep, rawId, 0).astype(np.uint32)
    pdigi = np.where(keep,
                     xx.astype(np.uint32)
                     | (yy.astype(np.uint32) << PACK_COLUMN_SHIFT)
                     | (np.minimum(adc, PACK_MAX_ADC).astype(np.uint32) << PACK_ADC_SHIFT),
                     0).astype(np.uint32)
    return xx, yy, adc, moduleInd, pdigi, rawIdArr


def calibrate(xx, yy, adc, moduleInd, c):
    """gpuCalibPixel::calibDigis, over every digi at once."""
    good = moduleInd != INV_ID
    mod = np.where(good, moduleInd, 0).astype(np.int64)

    first = c["rangeFirst"][mod]
    nCols = np.maximum(c["nCols"][mod], 1)
    lengthOfColumnData = (c["rangeLast"][mod] - first) // nCols
    offset = first + yy.astype(np.int64) * lengthOfColumnData + 2 * (xx.astype(np.int64) // c["rowsAveragedOver"])
    offset = np.clip(np.where(good, offset, 0), 0, c["pedestals"].size - 2)

    gainByte = c["pedestals"][offset]
    pedByte = c["pedestals"][offset + 1]
    dead = pedByte == c["deadFlag"]
    noisy = pedByte == c["noisyFlag"]

    pedestal = pedByte.astype(np.float32) * c["pedPrecision"] + c["minPed"]
    gain = gainByte.astype(np.float32) * c["gainPrecision"] + c["minGain"]

    isL1 = moduleInd < LAYER_1_MODULES
    conversion = np.where(isL1, VCAL_TO_ELECTRON_GAIN_L1, VCAL_TO_ELECTRON_GAIN).astype(np.float32)
    electrons = np.where(isL1, VCAL_TO_ELECTRON_OFFSET_L1, VCAL_TO_ELECTRON_OFFSET).astype(np.float32)

    vcal = adc.astype(np.float32) * gain - pedestal * gain
    calibrated = np.maximum(100, (vcal * conversion + electrons).astype(np.int32)).astype(np.uint16)

    bad = good & (dead | noisy)
    return (np.where(good & ~bad, calibrated, np.where(bad, 0, adc)).astype(np.uint16),
            np.where(bad, INV_ID, moduleInd).astype(np.uint16))


def module_boundaries(moduleInd, nWords):
    """countModules: where each module's pixels start.

    A module boundary is a valid pixel whose predecessor is in another module;
    the data has each module's pixels contiguous.  The C++ also refuses to
    cluster more than 4000 pixels of one module, which nothing in this data
    comes near -- if it ever did, the two would disagree, so it is an error
    rather than a silent difference.
    """
    valid = np.flatnonzero(moduleInd[:nWords] != INV_ID)
    if valid.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return valid, empty, np.zeros(0, dtype=np.uint32), empty, 0

    mod = moduleInd[valid].astype(np.int64)
    boundary = np.empty(valid.size, dtype=bool)
    boundary[0] = True
    boundary[1:] = mod[1:] != mod[:-1]
    firstPixel = valid[boundary]
    moduleIds = mod[boundary].astype(np.uint32)
    moduleOf = np.cumsum(boundary) - 1

    span = np.diff(np.append(firstPixel, nWords))
    if (span > MAX_PIX_IN_MODULE).any():
        raise RuntimeError(f"module with {int(span.max())} pixels, more than {MAX_PIX_IN_MODULE}")
    return valid, firstPixel, moduleIds, moduleOf, firstPixel.size


def sorted_pixels(valid, moduleOf, xx, yy):
    """The valid pixels by (module, column, row), and that key.

    Both labellings start here: the C++ bins the pixels of a module by column,
    and sorting on the three together is that for every module at once.  The
    key is packed into one ascending integer, which is what makes the pixels a
    labelling has to look at contiguous stretches of one array.
    """
    packed = (moduleOf * COL_STRIDE + yy[valid].astype(np.int64)) * ROW_STRIDE + xx[valid].astype(np.int64)
    sorter = np.argsort(packed, kind="stable")
    return valid[sorter], packed[sorter]


def number_clusters(label, valid, moduleOf, nModules, firstPixel, nWords):
    """The cluster each pixel belongs to, numbered as findClus numbers them.

    A cluster is labelled by its lowest pixel, so the pixels that are their own
    label are the clusters; counting those in index order gives each module's
    clusters the numbers the C++ scan gives them.
    """
    clus = np.arange(nWords, dtype=np.int32)
    isRoot = label == valid
    rank = np.empty(nWords, dtype=np.int64)
    rank[valid[isRoot]] = np.cumsum(isRoot)[isRoot] - 1
    counts = np.bincount(moduleOf[isRoot], minlength=nModules)
    firstOfModule = np.cumsum(counts) - counts
    clusterId = (rank[label] - firstOfModule[moduleOf]).astype(np.int32)

    # everything the module ranges cover is either a cluster or invalid; words
    # before the first module keep the index findClus never rewrote
    clus[firstPixel[0]:nWords] = -9999
    clus[valid] = clusterId
    return clus, counts


def charge_cut(adc, moduleInd, clus, valid, moduleOf, moduleIds, counts, nModules):
    """gpuClustering::clusterChargeCut, over every cluster at once."""
    base = np.concatenate(([0], np.cumsum(counts)))
    flat = base[moduleOf] + clus[valid]

    charge = np.zeros(int(base[-1]), dtype=np.int64)
    np.add.at(charge, flat, adc[valid].astype(np.int64))
    moduleOfCluster = np.repeat(np.arange(nModules), counts)
    cut = np.where(moduleIds[moduleOfCluster] < LAYER_1_MODULES, CHARGE_CUT_L1, CHARGE_CUT)
    ok = charge > cut

    # renumber the survivors of each module, which is an inclusive prefix
    # sum within the module
    inclusive = np.cumsum(ok)
    beforeModule = np.repeat(inclusive[base[:-1]] - ok[base[:-1]], counts)
    newId = inclusive - beforeModule  # 1-based among the survivors
    killed = ~ok

    survivors = newId[flat] - 1
    pixelKilled = killed[flat]
    clus = clus.copy()
    clus[valid] = np.where(pixelKilled, INV_ID, survivors)
    moduleInd = moduleInd.copy()
    moduleInd[valid] = np.where(pixelKilled, INV_ID, moduleInd[valid])

    survivorsPerModule = np.zeros(nModules, dtype=np.int64)
    np.maximum.at(survivorsPerModule, moduleOfCluster, np.where(ok, newId, 0))
    return clus, moduleInd, survivorsPerModule


def produce(module, event, eventSetup, components):
    """Everything a clusterizer module does except the labelling.

    `components` is the labelling: it takes the valid pixels and says, for each
    of them, the lowest pixel index of the cluster it belongs to.  That is the
    only thing the two modules do differently.
    """
    if module.conditions is None:
        module.conditions = cache_conditions(eventSetup)
    c = module.conditions

    words, feds = fed_words(event.get(module.raw), c["fedIds"])
    nWords = words.size

    xx, yy, adc, moduleInd, pdigi, rawIdArr = raw_to_digi(words, feds, c)
    adc, moduleInd = calibrate(xx, yy, adc, moduleInd, c)

    valid, firstPixel, moduleIds, moduleOf, nModules = module_boundaries(moduleInd, nWords)
    if nModules:
        label = components(valid, moduleOf, xx, yy, nWords)
        clus, counts = number_clusters(label, valid, moduleOf, nModules, firstPixel, nWords)
        clus, moduleInd, counts = charge_cut(
            adc, moduleInd, clus, valid, moduleOf, moduleIds, counts, nModules)
    else:
        clus = np.arange(nWords, dtype=np.int32)
        counts = np.zeros(0, dtype=np.int64)

    clusInModule = np.zeros(MAX_NUM_MODULES, dtype=np.uint32)
    if nModules:
        clusInModule[moduleIds] = counts.astype(np.uint32)

    # the hit offsets, capped per module and then in total, so that the rec hit
    # producer knows where each module's hits start
    clusModuleStart = np.zeros(MAX_NUM_MODULES + 1, dtype=np.uint32)
    clusModuleStart[1:] = np.cumsum(np.minimum(clusInModule, MAX_HITS_IN_MODULE))
    np.minimum(clusModuleStart, MAX_NUM_CLUSTERS, out=clusModuleStart)

    digis = event.allocate(module.digis, MAX_FED_WORDS)
    digis.setNModulesDigis(nModules, nWords)
    np.asarray(digis.xxSpan)[:] = xx
    np.asarray(digis.yySpan)[:] = yy
    np.asarray(digis.adcSpan)[:] = adc
    np.asarray(digis.moduleIndSpan)[:] = moduleInd
    np.asarray(digis.clusSpan)[:] = clus
    np.asarray(digis.pdigiSpan)[:] = pdigi
    np.asarray(digis.rawIdArrSpan)[:] = rawIdArr

    clusters = event.allocate(module.clusters, MAX_NUM_MODULES)
    moduleStart = np.asarray(clusters.moduleStartSpan)
    moduleStart[0] = nModules
    moduleStart[1 : 1 + nModules] = firstPixel
    np.asarray(clusters.clusInModuleSpan)[:] = clusInModule
    np.asarray(clusters.moduleIdSpan)[:nModules] = moduleIds
    np.asarray(clusters.clusModuleStartSpan)[:] = clusModuleStart
    clusters.setNClusters(int(clusModuleStart[MAX_NUM_MODULES]))
