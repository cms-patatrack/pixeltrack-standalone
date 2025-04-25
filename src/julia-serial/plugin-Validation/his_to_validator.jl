histos = Dict(
    "digi_n" => SimpleAtomicHisto(100, 0f0, 1f5),
    "digi_adc" => SimpleAtomicHisto(250, 0f0, 5f4),
    "module_n" => SimpleAtomicHisto(100, 1500f0, 2000f0),
    "cluster_n" => SimpleAtomicHisto(200, 5000f0, 25000f0),
    "cluster_per_module_n" => SimpleAtomicHisto(110, 0f0, 110f0),
    "hit_n" => SimpleAtomicHisto(200, 5000f0, 25000f0),
    "hit_lx" => SimpleAtomicHisto(200, -1f0, 1f0),
    "hit_ly" => SimpleAtomicHisto(800, -4f0, 4f0),
    "hit_lex" => SimpleAtomicHisto(100, 0f0, 5f-5),
    "hit_ley" => SimpleAtomicHisto(100, 0f0, 1f-4),
    "hit_gx" => SimpleAtomicHisto(200, -20f0, 20f0),
    "hit_gy" => SimpleAtomicHisto(200, -20f0, 20f0),
    "hit_gz" => SimpleAtomicHisto(600, -60f0, 60f0),
    "hit_gr" => SimpleAtomicHisto(200, 0f0, 20f0),
    "hit_charge" => SimpleAtomicHisto(400, 0f0, 4f6),
    "hit_sizex" => SimpleAtomicHisto(800, 0f0, 800f0),
    "hit_sizey" => SimpleAtomicHisto(800, 0f0, 800f0),
    "track_n" => SimpleAtomicHisto(150, 0f0, 15000f0),
    "track_nhits" => SimpleAtomicHisto(3, 3f0, 6f0),
    "track_chi2" => SimpleAtomicHisto(100, 0f0, 40f0),
    "track_pt" => SimpleAtomicHisto(400, 0f0, 400f0),
    "track_eta" => SimpleAtomicHisto(100, -3f0, 3f0),
    "track_phi" => SimpleAtomicHisto(100, -3.15f0, 3.15f0),
    "track_tip" => SimpleAtomicHisto(100, -1f0, 1f0),
    "track_tip_zoom" => SimpleAtomicHisto(100, -0.05f0, 0.05f0),
    "track_zip" => SimpleAtomicHisto(100, -15f0, 15f0),
    "track_zip_zoom" => SimpleAtomicHisto(100, -0.1f0, 0.1f0),
    "track_quality" => SimpleAtomicHisto(6, 0f0, 6f0),
    "vertex_n" => SimpleAtomicHisto(60, 0f0, 60f0),
    "vertex_z" => SimpleAtomicHisto(100, -15f0, 15f0),
    "vertex_chi2" => SimpleAtomicHisto(100, 0f0, 40f0),
    "vertex_ndof" => SimpleAtomicHisto(170, 0f0, 170f0),
    "vertex_pt2" => SimpleAtomicHisto(100, 0f0, 4000f0)
)

struct HistoValidator <: EDProducer
    digi_token::EDGetTokenT{SiPixelDigisSoA}
    cluster_token::EDGetTokenT{SiPixelClustersSoA}
    track_token::EDGetTokenT{TrackSOA}
    hit_token::EDGetTokenT{TrackingRecHit2DHeterogeneous}
    function HistoValidator(reg::ProductRegistry)
        new(consumes(reg,SiPixelDigisSoA),
        consumes(reg,SiPixelClustersSoA),consumes(reg,TrackSOA),consumes(reg,TrackingRecHit2DHeterogeneous))
    end
end
function produce(self::HistoValidator,i_event::Event,i_setup::EventSetup)
    digis = get(i_event,self.digi_token)
    clusters = get(i_event,self.cluster_token)
    n_digis = digis.n_digis_h
    n_modules = digis.n_modules_h
    n_clusters = clusters.nClusters_h
    hits = hist_view(get(i_event,self.hit_token))
    fill!(histos["digi_n"],n_digis)

    for i ∈ 1:n_digis
        fill!(histos["digi_adc"],digis.adc_d[i])
    end
    fill!(histos["module_n"],n_modules)
    fill!(histos["cluster_n"],n_clusters)

    for i ∈ 1:n_modules
        fill!(histos["cluster_per_module_n"],clusters.clus_in_module_d[i])
    end
    n_hits = hits.m_nHits
    fill!(histos["hit_n"],n_hits)
    # Loop through each hit and fill histograms
    for i in 1:n_hits
        fill!(histos["hit_lx"], hits.m_xl[i])
        fill!(histos["hit_ly"], hits.m_yl[i])
        fill!(histos["hit_lex"], hits.m_xerr[i])
        fill!(histos["hit_ley"], hits.m_yerr[i])
        fill!(histos["hit_gx"], hits.m_xg[i])
        fill!(histos["hit_gy"], hits.m_yg[i])
        fill!(histos["hit_gz"], hits.m_zg[i])
        fill!(histos["hit_gr"], hits.m_rg[i])
        fill!(histos["hit_charge"], hits.m_charge[i])
        fill!(histos["hit_sizex"], hits.m_xsize[i])
        fill!(histos["hit_sizey"], hits.m_ysize[i])
    end
    # tracks = get(i_event,self.track_token)
    # n_tracks = 0 
    # for i ∈ 1:stride_track(tracks)
    #     if tracks.n_hits[i] > 0 && tracks.quality[i] >= loose 
    #         n_tracks += 1
    #         fill!(histos["track_nhits"], tracks.nHits(i))
    #         fill!(histos["track_chi2"], tracks.chi2(i))
    #         fill!(histos["track_pt"], tracks.pt(i))
    #         fill!(histos["track_eta"], tracks.eta(i))
    #         fill!(histos["track_phi"], tracks.phi(i))
    #         fill!(histos["track_tip"], tracks.tip(i))
    #         fill!(histos["track_tip_zoom"], tracks.tip(i))
    #         fill!(histos["track_zip"], tracks.zip(i))
    #         fill!(histos["track_zip_zoom"], tracks.zip(i))
    #         fill!(histos["track_quality"], tracks.quality(i))
    #     end
    # end
    # fill!(histos["track_n"],n_tracks)
    # vertices = get(i_event,ZVertexSoA)
    # fill!(histos["vertex_n"],vertices.nv_final)

    # for i ∈ 1:vertices.nv_final
    #     fill!(histos["vertex_z"], vertices.zv[i])
    #     fill!(histos["vertex_chi2"], vertices.chi2[i])
    #     fill!(histos["vertex_ndof"], vertices.ndof[i])
    #     fill!(histos["vertex_pt2"], vertices.ptv2[i])
    # end
end

