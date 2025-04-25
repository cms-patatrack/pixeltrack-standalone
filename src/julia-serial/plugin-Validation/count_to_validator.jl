all_events::UInt32 = 0 
good_events::UInt32 = 0
using .Tracks:stride_track,n_hits_track
struct CountValidator <: EDProducer
    digi_cluster_count_token::EDGetTokenT{DigiClusterCount}
    track_count_token::EDGetTokenT{TrackCount}
    digi_token::EDGetTokenT{SiPixelDigisSoA}
    cluster_token::EDGetTokenT{SiPixelClustersSoA}
    track_token::EDGetTokenT{TrackSOA}
    vertex_token::EDGetTokenT{ZVertexSoA}
    vertex_count_token::EDGetTokenT{VertexCount}
    function CountValidator(reg::ProductRegistry)
        new(consumes(reg,DigiClusterCount),consumes(reg,TrackCount),consumes(reg,SiPixelDigisSoA),
        consumes(reg,SiPixelClustersSoA),consumes(reg,TrackSOA),consumes(reg,ZVertexSoA),consumes(reg,VertexCount))
    end
end

function produce(self::CountValidator,i_event::Event,i_setup::EventSetup)
    global all_events 
    global good_events 
    global counting
    track_tolerance = 0.012f0
    vertex_tolerance = 1
    buffer = IOBuffer()
    good_event = true 
    println(buffer,"Event ",i_event.eventId)
    digi_cluster_count = get(i_event,self.digi_cluster_count_token)
    track_count = get(i_event,self.track_count_token)
    digis = get(i_event,self.digi_token)
    clusters = get(i_event,self.cluster_token)
    tracks = get(i_event,self.track_token)
    vertices = get(i_event,self.vertex_token)
    vertex_count = get(i_event,self.vertex_count_token)
    if(digis.n_modules_h != digi_cluster_count.n_modules)
        println(buffer,"Number of modules is: ",digis.n_modules," Expected ",digi_cluster_count.n_modules)
        good_event = false 
    end

    if(digis.n_digis_h != digi_cluster_count.n_digis)
        println(buffer,"Number of digis is: ",digis.n_digis," Expected ",digi_cluster_count.n_digis)
        good_event = false 
    end

    if(clusters.nClusters_h != digi_cluster_count.n_clusters)
        println(buffer,"Number of digis is: ",digis.n_clusters," Expected ",digi_cluster_count.n_clusters)
        good_event = false 
    end


    n_tracks = 0
    sum_track_difference = 0
    for i ∈ 1:stride_track(tracks)
        if n_hits_track(tracks,i) > 0
            n_tracks += 1
        end
    end
    rel = abs(float(n_tracks - track_count.n_tracks) / track_count.n_tracks)
    println(n_tracks," ", track_count.n_tracks)
    if n_tracks != track_count.n_tracks 
        sum_track_difference += rel 
    end
    if rel >= track_tolerance
        println(buffer,"N(track) is ",n_tracks," expected ", track_count.n_tracks, ", relative difference ",rel)
        good_event = false
    end
    sum_vertex_difference = 0 
    diff = abs(Int(vertices.nv_final) - Int(vertex_count.vertices))
    if diff != 0
        sum_vertex_difference += diff
    end
    if diff >= vertex_tolerance
        println(buffer,"N(vertices) is ",vertices.nv_final," expected ", vertex_count.vertices, ", relative difference ",diff)
        good_event = false
    end


    all_events += 1
    if good_event
        good_events += 1
    else
        summary = String(take!(buffer))
        print(summary)
    end
end
add_plugin_module("CountValidator",x -> CountValidator(x))
