using .PluginFactory

mutable struct StreamSchedule
    registry::ProductRegistry
    source::Source
    event_setup::EventSetup
    stream_id::Int
    path::Vector{EDProducer}

    function StreamSchedule(
        reg::ProductRegistry,
        source::Source,
        event_setup::EventSetup,
        stream_id::Int,
        path::Vector{String}
    )
        reg = deepcopy(reg)  #important for creating a copy
        path_storage = Vector{EDProducer}()
        # print(source.raw_events)

        modInd = 1

        for name in path
            begin_module_construction(reg,modInd)
            push!(path_storage,create_plugin_module(name,reg))
            modInd += 1
        end

        return new(reg, source, event_setup, stream_id, path_storage)
    end
end


# function run_stream(ss::StreamSchedule)
#     Dagger.spawn_streaming() do
#         eventTask = Dagger.@spawn produce(ss.source, ss.stream_id, ss.registry)
#         # TaskClusterizer = Dagger.@spawn(eventTask) do
#         #     if event === nothing
#         #         @info "No more events to process. Stream $(ss.stream_id) exiting."
#         #         return Dagger.finish_streaming("Finished!")
#         #     end
#         #     produce(ss.path[1], event, ss.event_setup)
#         # end
#         eventTask_1 = Dagger.@spawn eventTask[1]
#         eventTask_2 = Dagger.@spawn eventTask[2]

#         TaskClusterizer = Dagger.@spawn produce(ss.path[1], eventTask_1, eventTask_2)
#         # TaskBeamspot = Dagger.@spawn produce(ss.path[2], event, ss.event_setup)
#         # TaskRecHit =  Dagger.@spawn begin
#         #     fetch(TaskClusterizer)  # Wait for Task A
#         #     fetch(TaskBeamspot)  # Wait for Task B
#         #     produce(ss.path[3], event, ss.event_setup)
#         # end
#         # #......
#     end
# end

function run_stream(ss::StreamSchedule)
    # print(ss.source.raw_events[1])
    # sleep(2)
    while true
        # print("in run_stream")
        
        event = produce(ss.source, ss.stream_id, ss.registry)
        if event === nothing
            @info "No more events to process. Stream $(ss.stream_id) exiting."
            break
        end
        # println("Event ID::", event.eventId, "Stream ID:: ", event.streamId)

        for i in 1:length(ss.path)
            produce(ss.path[i], event, ss.event_setup)
        end

    end
end




# tell dagger how many threads to use 
# fetch



