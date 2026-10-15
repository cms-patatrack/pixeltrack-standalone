findClus ${findClus * 32}
RawToDigi_kernel 512
kernel_connect_threads 64
kernel_connect_stride 4
getHits ${getHits * 32}
kernel_find_ntuplets ${kernel_find_ntuplets * 32}
fishbone_threads ${fishbone_threads * 32}
fishbone_stride ${fishbone_stride * 2}
clusterChargeCut ${clusterChargeCut * 32}
calibDigis 256
countModules 256
kernelLineFit3_threads ${kernelLineFit_threads * 32}
kernelLineFit4_threads ${kernelLineFit_threads * 32}
kernelLineFit5_threads ${kernelLineFit_threads * 32}
kernelFastFit3_threads 64
kernelFastFit4_threads 64
kernelFastFit5_threads 64
kernelFastFit3_blocks 384
kernelFastFit4_blocks 384
kernelFastFit5_blocks 384
kernel_fillHitDetIndices 128
finalizeBulk 128
kernel_earlyDuplicateRemover 128
kernel_countMultiplicity 128
kernel_fillMultiplicity 128
initDoublets 128
kernel_classifyTracks 64
kernel_fishboneCleaner 128
kernel_fastDuplicateRemover 64
