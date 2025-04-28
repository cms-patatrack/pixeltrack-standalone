using .ESPluginFactory

struct EventProcessor

    numberOfStreams::Int
    source::Source
    event_setup::EventSetup
    registry::ProductRegistry
    schedules::Vector{StreamSchedule}

    # Constructor
    function EventProcessor(numOfStreams::Int, path::Vector{String}, esproducers::Vector{String}, datadir::String, validation::Bool, max_events::Int)
        numberOfStreams = numOfStreams
        registry = ProductRegistry()
        source = Source(registry, datadir, validation, max_events)
        # print(source.raw_events)
        event_setup = EventSetup()
        for name in esproducers
            esp = create_plugin(name, datadir)
            produce(esp, event_setup)
        end

        schedules = Vector{StreamSchedule}()
        for i in 1:numberOfStreams
            push!(schedules, StreamSchedule(registry, source, event_setup, i, path))
        end

        new(numOfStreams, source, event_setup, registry, schedules)
    end
end



function run_processor(ev::EventProcessor)
    # tasks = [Dagger.@spawn run_stream(schedule) for schedule in ev.schedules]
    # # print(length(tasks))
    # # @info "Number of tasks created: $(length(tasks))"
    # # @info "Expected number of streams: $(ev.numberOfStreams)"
    # fetch.(tasks)
    # @threads for i in 1:ev.numberOfStreams
    #     run_stream(ev.schedules[i])
    # end
    @threads for i in 1:ev.numberOfStreams
        run_stream(ev.schedules[i])
    end

    # print(tasks)
    @info "All stream schedules have completed."
end

function warm_up(ev::EventProcessor, run_max_events::Int)
    @threads for i in 1:ev.numberOfStreams
        run_stream(ev.schedules[i])
    end
    ev.source.numEvents[] = 1
    ev.source.max_events = run_max_events
end
