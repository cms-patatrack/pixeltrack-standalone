module caConstants
export MAX_CELLS_PER_HIT, OuterHitOfCell, CellNeighbors, CellTracks, CellNeighborsVector, CellTracksVector, HitToTuple, TupleMultiplicity, OuterHitOfCellVector
export hindex_type
export MAX_NUM_OF_ACTIVE_DOUBLETS, MAX_NUM_OF_LAYER_PAIRS, MAX_NUM_OF_CONCURRENT_FITS
using ..histogram: OneToManyAssoc
using ..CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants: MAX_NUMBER_OF_HITS
using ..Patatrack: VecArray
using ..Patatrack: SimpleVector
using ..Patatrack: PreAllocMatrix
const MAX_NUM_TUPLES = 24 * 1024
const MAX_NUM_QUADRUPLETS = MAX_NUM_TUPLES
const MAX_NUM_OF_DOUBLETS = 512* 1024
const MAX_CELLS_PER_HIT = 128
const MAX_NUM_OF_ACTIVE_DOUBLETS = MAX_NUM_OF_DOUBLETS ÷ 8
const MAX_NUM_OF_LAYER_PAIRS = 20
const MAX_NUM_OF_LAYERS = 10
const MAX_TUPLES = MAX_NUM_TUPLES
const MAX_NUM_OF_CONCURRENT_FITS = 24 * 1024
const hindex_type = UInt16
const tindex_type = UInt16
const CellNeighbors = VecArray{UInt32,36}
const CellTracks = VecArray{tindex_type,48}
const CellNeighborsVector = PreAllocMatrix{UInt32}
const CellTracksVector = PreAllocMatrix{tindex_type}
const OuterHitOfCell = VecArray{UInt32,MAX_CELLS_PER_HIT}
const OuterHitOfCellVector = PreAllocMatrix{UInt32}
# const TuplesContainer = OneToManyAssoc{hindex_type,MAX_TUPLES,5*MAX_TUPLES}
const HitToTuple = OneToManyAssoc{tindex_type,MAX_NUMBER_OF_HITS,4 * MAX_TUPLES}
const TupleMultiplicity = OneToManyAssoc{tindex_type,8,MAX_TUPLES} # eg: 3 tracks (4 hits each), 2 tracks (3 hits), 1 track (2 Hits)=> [0, 1, 3]  number of tracks that have i hits = off[i+1] - off[i]
end
