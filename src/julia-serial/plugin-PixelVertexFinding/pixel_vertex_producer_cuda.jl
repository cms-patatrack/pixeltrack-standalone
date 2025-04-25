using .gpuVertexFinder:Producer,make
using .VertexSOA:ZVertexSoA
struct PixelVertexProducerCUDA <: EDProducer
    token_cpu_track::EDGetTokenT{TrackSOA}
    token_cpu_vertex::EDPutTokenT{ZVertexSoA}
    m_pt_min::Float32
    m_gpu_algo::Producer
    function PixelVertexProducerCUDA(reg::ProductRegistry)
        token_cpu_track= consumes(reg, TrackSOA)
        token_cpu_vertex = produces(reg, ZVertexSoA)
        m_gpu_algo = Producer(true,   # oneKernel
                              true,   # useDensity
                              false,  # useDBSCAN
                              false,  # useIterative
                              2,      # minT
                              0.07,   # eps
                              0.01,   # errmax
                              9 )    # chi2max
        new(token_cpu_track, token_cpu_vertex,0.5,m_gpu_algo)
    end
    
end

function produce(self::PixelVertexProducerCUDA,i_event::Event,es::EventSetup)
    tracks = get(i_event,self.token_cpu_track)
    vertices = make(self.m_gpu_algo,tracks,self.m_pt_min)
    emplace(i_event,self.token_cpu_vertex,vertices)
end

add_plugin_module("PixelVertexProducerCUDA",x -> PixelVertexProducerCUDA(x))