from std.sys import is_defined

from MojoSerial.CUDACore.VecArray import VecArray
from MojoSerial.CUDACore.SimpleVector import SimpleVector
from MojoSerial.CUDACore.HistoContainer import OneToManyAssoc
from MojoSerial.CUDADataFormats.GPUClusteringConstants import (
    GPUClusteringConstants,
)
from MojoSerial.MojoBridge.DTypes import DType


@parameter
def maxNumberOfTuples() -> UInt32:
    if is_defined["ONLY_PHICUT"]():
        return 48 * 1024
    if is_defined["GPU_SMALL_EVENTS"]():
        return 3 * 1024
    return 24 * 1024


@parameter
def maxNumberOfQuadruplets() -> UInt32:
    return maxNumberOfTuples()


@parameter
def maxNumberOfDoublets() -> UInt32:
    if is_defined["ONLY_PHICUT"]():
        return 2 * 1024 * 1024
    if is_defined["GPU_SMALL_EVENTS"]():
        return 128 * 1024
    return 512 * 1024


@parameter
def maxCellsPerHit() -> UInt32:
    if is_defined["ONLY_PHICUT"]():
        return 8 * 128
    if is_defined["GPU_SMALL_EVENTS"]():
        return 128 // 2
    return 128


@parameter
def maxNumOfActiveDoublets() -> UInt32:
    return maxNumberOfDoublets() // 8


@parameter
def maxNumberOfLayerPairs() -> UInt32:
    return 20


@parameter
def maxNumberOfLayers() -> UInt32:
    return 10


@parameter
def maxTuples() -> UInt32:
    return maxNumberOfTuples()


comptime _MaxCellsPerHit: Int = Int(maxCellsPerHit())

comptime _MaxNumberOfTuples: UInt32 = maxNumberOfTuples()

comptime _CellNeighborsCapacity: Int = (
    36 if not is_defined["ONLY_PHICUT"]() else 64
)

comptime _CellTracksCapacity: Int = 48 if not is_defined["ONLY_PHICUT"]() else 64

comptime hindex_type = UInt16
comptime tindex_type = UInt16


comptime CellNeighbors = VecArray[UInt32, "CellNeighbors", _CellNeighborsCapacity]
comptime CellTracks = VecArray[tindex_type, "CellTracks", _CellTracksCapacity]

comptime CellNeighborsVector = SimpleVector[CellNeighbors, "CellNeighborsVector"]
comptime CellTracksVector = SimpleVector[CellTracks, "CellTracksVector"]

comptime OuterHitOfCell = VecArray[UInt32, "OuterHitOfCell", _MaxCellsPerHit]
comptime TuplesContainer = OneToManyAssoc[
    DType.uint16, _MaxNumberOfTuples, 5 * _MaxNumberOfTuples
]
comptime HitToTuple = OneToManyAssoc[
    DType.uint16, GPUClusteringConstants.maxNumberOfHits, 4 * _MaxNumberOfTuples
]
comptime TupleMultiplicity = OneToManyAssoc[
    DType.uint16, 8, _MaxNumberOfTuples
]
