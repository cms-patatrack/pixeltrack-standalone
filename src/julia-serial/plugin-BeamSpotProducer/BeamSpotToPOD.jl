using .BeamSpotPOD_h
using .PluginFactory

struct BeamSpotToPOD <: EDProducer
    bsPutToken_::EDPutTokenT{BeamSpotPOD}

    function BeamSpotToPOD(reg::ProductRegistry)
        new(produces(reg,BeamSpotPOD))
    end
end

function produce(bs::BeamSpotToPOD , iEvent::Event, iSetup::EventSetup)
    emplace(iEvent,bs.bsPutToken_,get(iSetup,BeamSpotPOD))
end

add_plugin_module("BeamSpotToPOD",x -> BeamSpotToPOD(x))



