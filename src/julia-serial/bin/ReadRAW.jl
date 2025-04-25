
"""
    readall(io::IOStream)

Read data of one fed from a stream of data in the patatrack raw data format.

See also [readall](@ref).
"""
function readfed(io::IOStream)
    fedid = read(io, Int32)
    fedsize = read(io, Int32)
    feddata = read(io, fedsize)
    return FedRawData(fedid, feddata)
end

"""
    readevent(io::IOStream)

Read one event from a stream of data in the patatrack raw data format.

See also [readall](@ref).
"""
function readevent(io::IOStream)
    nfeds = read(io, Int32)
    collection::FedRawDataCollection = FedRawDataCollection()
    for i ∈ 1:nfeds
        raw_data = readfed(io)
        collection.data[raw_data.fedid] = raw_data
    end
    return collection
end

"""
    readall(io::IOStream)

Read all events from a stream of data in the patatrack raw data format.
"""
function readall(io::IOStream, vertex_io, track_io, digi_io, digi_cluster_count_v, track_count_v, vertex_count_v, validation::Bool)
    events = Vector{FedRawDataCollection}()

    if validation
        # Open all required files once
        while !eof(io)
            # Read and store event
            push!(events, readevent(io))

            # Read corresponding DigiClusterCount
            nm = read(digi_io, UInt32)
            nd = read(digi_io, UInt32)
            nc = read(digi_io, UInt32)
            push!(digi_cluster_count_v, DigiClusterCount(nm, nd, nc))

            # Read corresponding TrackCount
            nt = read(track_io, UInt32)
            push!(track_count_v, TrackCount(nt))

            # Read corresponding VertexCount
            nv = read(vertex_io, UInt32)
            push!(vertex_count_v, VertexCount(nv))
        end
    else
        while !eof(io)
            push!(events, readevent(io))
        end
    end

    return events
end
