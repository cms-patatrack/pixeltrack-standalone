using .dataFormats

mutable struct Source

    raw_events::Vector{FedRawDataCollection}
    numEvents::Atomic{Int}
    rawToken::EDPutTokenT{FedRawDataCollection}
    digiClusterToken_::EDPutTokenT{DigiClusterCount}
    trackToken_::EDPutTokenT{TrackCount}
    vertexToken_::EDPutTokenT{VertexCount}

    digi_cluster_count_v::Vector{DigiClusterCount}
    track_count_v::Vector{TrackCount}
    vertex_count_v::Vector{VertexCount}

    validation::Bool
    max_events::Int


    function Source(reg::ProductRegistry, dataDir::String, validation::Bool, max_events::Int)
        digiClusterToken_ = EDPutTokenT{DigiClusterCount}()
        trackToken_ = EDPutTokenT{TrackCount}()
        vertexToken_ = EDPutTokenT{VertexCount}()
        if (validation)
            digiClusterToken_ = produces(reg, DigiClusterCount)
            trackToken_ = produces(reg, TrackCount)
            vertexToken_ = produces(reg, VertexCount)
        end

        rawToken = produces(reg, FedRawDataCollection)
        rawFilePath = joinpath(dataDir, "raw.bin")

        verticesFilePath = joinpath(dataDir, "vertices.bin")
        tracksFilePath = joinpath(dataDir, "tracks.bin")
        digiclusterFilePath = joinpath(dataDir, "digicluster.bin")
        digi_cluster_count_v = DigiClusterCount[]
        track_count_v = TrackCount[]
        vertex_count_v = VertexCount[]
        if (validation)
            raw_events = readall(open(rawFilePath), open(verticesFilePath), open(tracksFilePath), open(digiclusterFilePath), digi_cluster_count_v, track_count_v, vertex_count_v, validation) # Reads 1000 event 
        else
            raw_events = readall(open(rawFilePath), open(verticesFilePath), open(tracksFilePath), open(digiclusterFilePath), nothing, nothing, nothing, validation) # Reads 1000 event 
        end

        return new(raw_events, Atomic{Int}(1), rawToken, digiClusterToken_, trackToken_, vertexToken_, digi_cluster_count_v, track_count_v, vertex_count_v, validation, max_events)
    end



end

function produce(src::Source, streamId::Int, reg::ProductRegistry)
    if src.numEvents.value > src.max_events
        return nothing
    end

    iev = (atomic_add!(src.numEvents, 1) - 1) % length(src.raw_events) + 1
    # println("Taking an Event ", iev)
    # print(src.raw_events)


    # if old >= src.maxEvents
    #     src.shouldStop = true
    #     atomic_sub!(src.numEvents, 1)
    #     return nothing
    # end
    ev = Event(streamId, iev, reg)
    emplace(ev, src.rawToken, src.raw_events[iev])
    if (src.validation)
        emplace(ev, src.digiClusterToken_, src.digi_cluster_count_v[iev])
        emplace(ev, src.trackToken_, src.track_count_v[iev])
        emplace(ev, src.vertexToken_, src.vertex_count_v[iev])
    end

    # if src.validation
    #     ev.reg[src.digiClusterToken] = src.digiclusters[index]
    #     ev.reg[src.trackToken] = src.tracks[index]
    #     ev.reg[src.vertexToken] = src.vertices[index]
    # end

    return ev
end
