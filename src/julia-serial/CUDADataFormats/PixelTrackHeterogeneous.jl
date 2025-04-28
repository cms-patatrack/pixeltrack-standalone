module Tracks

export TrackSOA, n_hits_track, stride_track
using ..Patatrack: Quality
using ..histogram: OneToManyAssoc, size
using ..eigenSOA: ScalarSOA
using ..CUDADataFormatsTrackTrajectoryStateSOA_H: TrajectoryStateSoA, copyFromCircle!
const hindex_type = UInt16

struct TrackSOAT{S,many}
    m_quality::ScalarSOA{Quality,S}
    chi2::ScalarSOA{Float32,S}
    eta::ScalarSOA{Float32,S}
    pt::ScalarSOA{Float32,S}
    hit_indices::OneToManyAssoc{hindex_type,S,many}
    det_indices::OneToManyAssoc{hindex_type,S,many}
    m_nTracks::UInt32
    stateAtBS::TrajectoryStateSoA

    # Constructor
    function TrackSOAT{S,many}() where {S,many}
        new(
            ScalarSOA{Quality,S}(),
            ScalarSOA{Float32,S}(),
            ScalarSOA{Float32,S}(),
            ScalarSOA{Float32,S}(),
            OneToManyAssoc{hindex_type,S,many}(),
            OneToManyAssoc{hindex_type,S,many}(),
            0,
            TrajectoryStateSoA(S)
        )
    end
end

function hit_indices(tracks::TrackSOAT)
    return tracks.hit_indices
end

function charge(track::TrackSOAT, i::Int)::Float32
    return copysign(1.0f0, track.stateAtBS.state[i][3])
end

function phi(track::TrackSOAT, i::Int)::Float32
    return track.stateAtBS.state[i][1]
end

function tip(track::TrackSOAT, i::Integer)::Float32
    return track.stateAtBS.state[i,2]
end

function zip(track::TrackSOAT, i::Integer)::Float32
    return track.stateAtBS.state[i,5]
end

const MAX_NUMBER = 32 * 1024



const TrackSOA = TrackSOAT{MAX_NUMBER,5 * MAX_NUMBER}
const HitContainer = OneToManyAssoc{hindex_type,MAX_NUMBER,5 * MAX_NUMBER}

n_hits_track(self::TrackSOAT{MAX_NUMBER,5 * MAX_NUMBER}, i::Integer) = size(self.det_indices, i)
stride_track(::TrackSOAT{S,many}) where {S,many} = S

end
