# module gpuConfig

# if isdefined(Main, :__CUDA_ARCH__)
#     if !isdefined(Main, :GPU_DEBUG)
#         if !isdefined(Main, :NDEBUG) || !NDEBUG
#             const NDEBUG = true
#         end
#     else
#         if isdefined(Main, :NDEBUG) && NDEBUG
#             const NDEBUG = false
#         end
#     end
# end

# end # module gpuConfig