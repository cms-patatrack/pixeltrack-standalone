struct CAHitNtuplet <: EDProducer
    token_hit_cpu::EDGetTokenT{TrackingRecHit2DHeterogeneous}
    token_track_cpu::EDPutTokenT{TrackSOA}
    gpu_algo::CAHitNtupletGeneratorOnGPU
    function CAHitNtuplet(reg::ProductRegistry)
        new(consumes(reg, TrackingRecHit2DHeterogeneous), produces(reg, TrackSOA), CAHitNtupletGeneratorOnGPU())
    end
end

function produce(self::CAHitNtuplet, i_event::Event, i_setup::EventSetup)
    bf = 0.0114256972711507 # 1/fieldInGeV
    hits = get(i_event, self.token_hit_cpu)
    # println(hits.m_nHits)
    tracks = make_tuples(self.gpu_algo, hits, bf)
    emplace(i_event, self.token_track_cpu, tracks)
end


add_plugin_module("CAHitNtupletCUDA", x -> CAHitNtuplet(x))
