module Patatrack
import Base.length
using Printf
using StaticArrays: MArray
using LinearAlgebra
# using Gtk4
using Base
using Base.Threads

export FedRawDataCollection, FedRawData

export SiPixelFedCablingMapGPU

export readall, readevent, readfed
export EventSetup
export SiPixelFedCablingMapGPUWrapperESProducer
export SiPixelGainCalibrationForHLTGPUESProducer
export produce
export SiPixelGainForHLTonGPU
export MAX_SIZE
export RANGE_COUNT
export RangeAndCols, DecodingStructure
export SiPixelRawToClusterCUDA
export PixelErrorCompact, PixelFormatterErrors
export has_quality, SiPixelFedCablingMapGPUWrapper
export get_cpu_product
export get_mod_to_unp_all
export fed_ids
export ErrorChecker
export make_clusters
export get_results
export FedData
export data
export check_crc
export check_trailer
export initialize_word_fed
export AverageGeometry
export ProductRegistry
export Event, EDPutTokenT, produces, emplace, begin_module_construction

export number_of_ladders_in_barrel

export local_x
export local_y
export is_big_pix_y
export is_big_pix_x

export last_row_in_module
export last_col_in_module

export x_offset
export y_offset

export BeamSpotPOD
export SOAFrame
export ParamsOnGPU
export position_corr

export errorFromDB

export PixelCPEFastESProducer
export BeamSpotESProducer
export BeamSpotToPOD
export SiPixelRecHitCUDA

export x_global
export y_global
export z_global

export set_x_global
export set_y_global
export set_z_global

export CAHitNtuplet

export EventProcessor
export Source
export run_processor
export DigiClusterCount
export TrackCount
export VertexCount
export CountValidator
export HistoValidator
export warm_up
export n_hits
include("../CUDACore/simple_matrix.jl")
include("../Framework/ESPluginFactory.jl")
include("../DataFormats/track_count.jl")
include("../DataFormats/vertex_count.jl")
include("../DataFormats/digi_cluster_count.jl")
include("../CUDACore/vec_array.jl")
include("../CUDACore/simple_vector.jl")
include("../CondFormats/si_pixel_fed_cabling_map_gpu.jl")
include("../CondFormats/si_pixel_fed_cabling_map_gpu_wrapper.jl")
include("../CondFormats/si_pixel_fed_ids.jl")
include("../CUDACore/cuda_assert.jl")
include("../CondFormats/si_pixel_gain_for_hlt_on_gpu.jl")
include("../CondFormats/si_pixel_gain_calibration_for_hlt_gpu.jl")
include("../CUDACore/cudaCompat.jl")
include("../CUDACore/cudastdAlgorithm.jl")
include("../CUDACore/prefix_scan.jl")
include("../CUDACore/hist_to_container.jl")
include("../CUDADataFormats/gpu_clustering_constants.jl")
include("../CUDADataFormats/SiPixelClusterSoA.jl")
include("../DataFormats/SiPixelRawDataError.jl")
include("../DataFormats/PixelErrors.jl")
include("../CUDADataFormats/SiPixelDigiErrorsSoA.jl")
include("../CUDADataFormats/SiPixelDigisSoA.jl")
include("../DataFormats/data_formats.jl")
include("../DataFormats/fed_header.jl")
include("../DataFormats/fed_trailer.jl")
include("../Framework/EDTokens.jl")
include("../Framework/ProductRegistry.jl")
include("../Framework/PluginFactory.jl")
include("../Framework/EandES.jl")
include("../Framework/ESProducer.jl")
include("../Framework/EDProducer.jl")
include("../Geometry/phase1PixelTopology.jl")
include("../plugin-SiPixelClusterizer/adc_threshold.jl")
include("../plugin-SiPixelClusterizer/Constants.jl")
include("../plugin-SiPixelClusterizer/ErrorChecker.jl")
include("../plugin-SiPixelClusterizer/gpu_calib_pixel.jl")
include("../plugin-SiPixelClusterizer/gpu_cluster_charge_cut.jl")
include("../plugin-SiPixelClusterizer/gpu_clustering.jl")
include("../plugin-SiPixelClusterizer/SiPixelFedCablingMapGPUWrapperESProducer.jl")
include("../plugin-SiPixelClusterizer/SiPixelGainCalibrationForHLTGPUESProducer.jl")
include("../plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.jl")
include("../plugin-SiPixelClusterizer/SiPixelRawToClusterCUDA.jl")
# include("../plugin-SiPixelClusterizer/testClustering.jl")
include("../bin/ReadRAW.jl")
include("../DataFormats/BeamSpotPOD.jl")
include("../plugin-BeamSpotProducer/BeamSpotToPOD.jl")
include("../plugin-BeamSpotProducer/BeamSpotESProducer.jl")

# include("../CUDADataFormats/HeterogeneousSoA.jl")
include("../DataFormats/SOARotation.jl")


include("../CondFormats/pixelCPEforGPU.jl")

include("../CUDADataFormats/TrackingRecHit2DSOAView.jl")
include("../CUDADataFormats/TrackingRecHit2DHeterogeneous.jl")
include("../DataFormats/approx_atan2.jl")

include("../plugin-SiPixelRecHits/gpuPixelRecHits.jl")

include("../CondFormats/PixelCPEFast.jl")

include("../plugin-SiPixelRecHits/PixelCPEFastESProducer.jl")

include("../plugin-SiPixelRecHits/PixelRecHits.jl")
include("../plugin-SiPixelRecHits/SiPixelRecHitCUDA.jl")
include("../plugin-PixelTriplets/ca_constants.jl")
include("../plugin-PixelTriplets/fit_result.jl")
include("../plugin-PixelTriplets/circle_eq.jl")
include("../CUDADataFormats/track_quality.jl")
include("../CUDACore/eigen_soa.jl")
include("../plugin-PixelTriplets/fit_utils.jl")
include("../plugin-PixelTriplets/cholesky_inversion.jl")
include("../plugin-PixelTriplets/broken_line.jl")
include("../CUDADataFormats/TrajectoryStateSOA.jl")
include("../CUDADataFormats/PixelTrackHeterogeneous.jl")
include("../plugin-PixelTriplets/gpu_ca_cell.jl")
include("../plugin-PixelTriplets/gpu_pixel_doublets.jl")
include("../plugin-PixelTriplets/ca_hit_ntuplet_generator_kernels_impl.jl")
include("../plugin-PixelTriplets/ca_hit_ntuplet_generator_kernels.jl")

include("../plugin-PixelTriplets/helix_fit_on_gpu.jl")
include("../plugin-PixelTriplets/broken_line_fit_on_gpu.jl")
include("../plugin-PixelTriplets/ca_hit_ntuplet_generator.jl")
include("../plugin-PixelTriplets/ca_hit_ntuplet.jl")
include("../plugin-Validation/simple_atomic_histo.jl")
include("../plugin-Validation/his_to_validator.jl")

include("../Framework/Source.jl")
include("../Framework/StreamSchedule.jl")
include("../Framework/EventProcessor.jl")

include("../CUDADataFormats/z_vertex_soa.jl")
include("../plugin-PixelVertexFinding/gpu_vertex_finder.jl")
include("../plugin-PixelVertexFinding/pixel_vertex_producer_cuda.jl")

include("../plugin-Validation/count_to_validator.jl")
include("entrypoints.jl")
end