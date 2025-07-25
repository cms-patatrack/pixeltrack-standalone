using .CUDADataFormatsSiPixelDigiInterfaceSiPixelDigisSoA
using .CUDADataFormatsSiPixelClusterInterfaceSiPixelClustersSoA
using .CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
using .CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
using .RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
using .PluginFactory

# more includes are missing (waiting on files to be done)

struct SiPixelRecHitCUDA <: EDProducer
    tBeamSpot::EDGetTokenT{BeamSpotPOD}
    token::EDGetTokenT{SiPixelClustersSoA}
    tokenDigi::EDGetTokenT{SiPixelDigisSoA}
    tokenHit::EDPutTokenT{TrackingRecHit2DHeterogeneous}

    function SiPixelRecHitCUDA(reg::ProductRegistry)
        tBeamSpot= consumes(reg, BeamSpotPOD)
        token = consumes(reg, SiPixelClustersSoA)
        tokenDigi = consumes(reg, SiPixelDigisSoA)
        tokenHit = produces(reg, TrackingRecHit2DHeterogeneous)
        new(tBeamSpot, token, tokenDigi, tokenHit)
    end
end

function produce(self::SiPixelRecHitCUDA,iEvent::Event, es::EventSetup)
    fcpe = get(es, PixelCPEFast)
    
    clusters = get(iEvent, self.token)
    
    digis = get(iEvent, self.tokenDigi)
    
    bs = get(iEvent, self.tBeamSpot)
    
    nHits = nClusters(clusters)
    # println(nHits, "id: $(iEvent.eventId)")
    if nHits >= max_hits()
        println("Clusters/Hits Overflow ",nHits," >= ", TrackingRecHit2DSOAView::maxHits())
    end
    #makeHits(digis, clusters, bs, getCPUProduct(fcpe))
    emplace(iEvent, self.tokenHit, makeHits(digis, clusters, bs, getCPUProduct(fcpe)))
    
end


add_plugin_module("SiPixelRecHitCUDA",x -> SiPixelRecHitCUDA(x))
