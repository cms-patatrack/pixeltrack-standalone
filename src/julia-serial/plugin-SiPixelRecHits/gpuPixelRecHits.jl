module RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelRecHits_h

using ..BeamSpotPOD_h: BeamSpotPOD
using ..Geometry_TrackerGeometryBuilder_phase1PixelTopology_h.phase1PixelTopology
# using ..gpuConfig
using ..CUDADataFormatsSiPixelClusterInterfaceSiPixelClustersSoA
using ..CUDADataFormatsSiPixelDigiInterfaceSiPixelDigisSoA
using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
using ..PixelGPU_h
using ..SOA_h
using ..DataFormatsMathAPPROX_ATAN2_H

export getHits
""" getHits function

 Processes pixel hit data from clusters and digis, adjusts for the beam spot position, 
    and calculates hit positions and errors. This function operates in iterations over the modules and clusters.

    **Inputs**:
    - `cpeParams::ParamsOnGPU`: Parameters for the cluster position estimation on GPU.
    - `bs::BeamSpotPOD`: Beam spot position and spread.
    - `pdigis::DeviceConstView`: Device view for accessing pixel digi data.
    - `numElements::Integer`: Number of elements in the digi data.
    - `pclusters::DeviceConstView`: Device view for accessing pixel cluster data.
    - `phits::Vector{TrackingRecHit2DSOAView}`: Vector of tracking hit views where results will be stored.

    **Outputs**:
    - `phits` is updated with the calculated hit positions, charges, sizes, and errors.

"""
function getHits(cpeParams::ParamsOnGPU, 
                 bs::BeamSpotPOD, 
                 pdigis::CUDADataFormatsSiPixelDigiInterfaceSiPixelDigisSoA.DeviceConstView,
                 numElements::Integer,
                 pclusters::CUDADataFormatsSiPixelClusterInterfaceSiPixelClustersSoA.DeviceConstView,
                 phits::TrackingRecHit2DSOAView)


        hits = phits
        digis = pdigis
        clusters = pclusters
#
       # write(file, "FOR 1 EVENT THIS IS WHAT's HAPPENING:\n##############################################\n")

        agc = average_geometry(hits)
        ag = averageGeometry(cpeParams)

   #     write(file, "FOR I FROM 1 to $(number_of_ladders_in_barrel)\n")

        for il in 1:number_of_ladders_in_barrel
       #     write(file, "agc.ladderZ[$il] = $(ag.ladderZ[il] - bs.z)\n")
            agc.ladderZ[il] = ag.ladderZ[il] - bs.z
       #     write(file, "agc.ladderX[$il] = $(ag.ladderX[il] - bs.z)\n")
            agc.ladderX[il] = ag.ladderX[il] - bs.x
       #     write(file, "agc.ladderY[$il] = $(ag.ladderY[il] - bs.z)\n")
            agc.ladderY[il] = ag.ladderY[il] - bs.y
       #     write(file, "agc.ladderR[$il] = $(sqrt(agc.ladderX[il] * agc.ladderX[il] + agc.ladderY[il] * agc.ladderY[il]))\n")
            agc.ladderR[il] = sqrt(agc.ladderX[il] * agc.ladderX[il] + agc.ladderY[il] * agc.ladderY[il])
       #     write(file, "agc.ladderMinZ[$il] = $(ag.ladderMinZ[il] - bs.z)\n")
            agc.ladderMinZ[il] = ag.ladderMinZ[il] - bs.z
       #     write(file, "agc.ladderMaxZ[$il] = $(ag.ladderMaxZ[il] - bs.z)\n")
            agc.ladderMaxZ[il] = ag.ladderMaxZ[il] - bs.z
        end
   #     write(file, "agc.endCapZ[0] = $(ag.endCapZ[1] - bs.z)\n")
        agc.endCapZ[1] = ag.endCapZ[1] - bs.z
   #     write(file, "agc.endCapZ[1] = $(ag.endCapZ[2] - bs.z)\n")
        agc.endCapZ[2] = ag.endCapZ[2] - bs.z

   #     write(file, "##############################################\n")



        InvId = 9999
        MaxHitsInIter = PixelGPU_h.MaxHitsInIter

        clusParams = ClusParamsT{PixelGPU_h.MaxHitsInIter}()

        firstModule = 1
     #    write(file, "firstModule: $firstModule\n")

        endModule = module_start(clusters, 1)
        #print(endModule)
     #    write(file, "endModule: $endModule\n")
     #    write(file, "##############################################\n")
     #    write(file, "FOR module 1 to $(endModule )\n")

        for mod in firstModule:endModule
            #write(file, "me = $(module_id(clusters, mod))\n")
            me = module_id(clusters, mod)
            nclus = clus_in_module(clusters, UInt32(me + 1))
            #write(file, "nclus = $(nclus)\n")
            
            if 0 == nclus
                continue
            end
            
        endClus = nclus

        #write(file, "FOR startClus 1 to $(nclus) incrementing by $MaxHitsInIter\n")

            for startClus in 1:MaxHitsInIter:(endClus)
                first = module_start(clusters, mod + 1)
               #  write(file, "first: $first\n")


                nClusInIter = min(MaxHitsInIter, nclus - startClus + 1)
               #  write(file, "nClusInIter: $nClusInIter\n")
                lastClus = startClus - 1 + nClusInIter
               #  write(file, "lastClus: $lastClus\n")
                @assert nClusInIter <= nclus
                @assert nClusInIter > 0
                @assert lastClus <= nclus
                @assert nclus > MaxHitsInIter || (1 == startClus && nClusInIter == nclus && lastClus == nclus)
                
               #  write(file, "##############################################\n")
               #  write(file, "FOR ic 1 to $(nClusInIter )\n")

                for ic in 1:nClusInIter
                    clusParams.minRow[ic] = UInt32(typemax(UInt32))
                    #write(file, "clusParams.minRow[$ic] = $(clusParams.minRow[ic])\n")
                    
                    clusParams.maxRow[ic] = zero(UInt32)
                    #write(file, "clusParams.maxRow[$ic] = $(clusParams.maxRow[ic])\n")
                    
                    clusParams.minCol[ic] = UInt32(typemax(UInt32))
                    #write(file, "clusParams.minCol[$ic] = $(clusParams.minCol[ic])\n")
                    
                    clusParams.maxCol[ic] = zero(UInt32)
                    #write(file, "clusParams.maxCol[$ic] = $(clusParams.maxCol[ic])\n")
                    
                    clusParams.charge[ic] = zero(UInt32)
                    #write(file, "clusParams.charge[$ic] = $(clusParams.charge[ic])\n")
                    
                    clusParams.Q_f_X[ic] = zero(UInt32)
                    #write(file, "clusParams.Q_f_X[$ic] = $(clusParams.Q_f_X[ic])\n")
                    
                    clusParams.Q_l_X[ic] = zero(UInt32)
                    #write(file, "clusParams.Q_l_X[$ic] = $(clusParams.Q_l_X[ic])\n")
                    
                    clusParams.Q_f_Y[ic] = zero(UInt32)
                    #write(file, "clusParams.Q_f_Y[$ic] = $(clusParams.Q_f_Y[ic])\n")
                    
                    clusParams.Q_l_Y[ic] = zero(UInt32)
                    #write(file, "clusParams.Q_l_Y[$ic] = $(clusParams.Q_l_Y[ic])\n")
                end
                
                #write(file,"FOR i in $first to $numElements \n")

                for i in first:numElements
                    id = module_ind(digis, i)
                    #write(file, "id = $id\n")
                    if id == InvId
                        continue
                    end
                    if id != me
                        break
                    end
                    cl = clus(digis, i)
                    #write(file, "cl = $cl\n")
                    
                    if cl < startClus || cl > lastClus
                        continue
                    end
                    
                    x = xx(digis, i)
                    #write(file, "x = $x\n")
                    
                    y = yy(digis, i)
                    #write(file, "y = $y\n")
                    
                    cl = cl - startClus + 1
                    @assert cl >= 1 
                    @assert cl <= MaxHitsInIter  # will verify later
                    
                    if clusParams.minRow[cl] > x
                        clusParams.minRow[cl] = x
                    end
                    #write(file, "clusParams.minRow[$cl] = $(clusParams.minRow[cl])\n")
                    
                    if clusParams.maxRow[cl] < x
                        clusParams.maxRow[cl] = x
                    end
                    #write(file, "clusParams.maxRow[$cl] = $(clusParams.maxRow[cl])\n")
                    
                    if clusParams.minCol[cl] > y
                        clusParams.minCol[cl] = y
                    end
                    #write(file, "clusParams.minCol[$cl] = $(clusParams.minCol[cl])\n")
                    
                    if clusParams.maxCol[cl] < y
                        clusParams.maxCol[cl] = y
                    end
                    #write(file, "clusParams.maxCol[$cl] = $(clusParams.maxCol[cl])\n")
                end

                pixmx = typemax(UInt16)
           #     write(file,"##################################\n")
           #     write(file,"pixmx: $pixmx\n")
           #     write(file,"FOR i IN $first to $numElements\n")
                for i in first:numElements
                    id = module_ind(digis, i)
               #     write(file, "id = $id\n")
                    
                    if id == InvId
                        continue
                    end
                    
                    if id != me
                        break
                    end
                    
                    cl = clus(digis, i)
               #     write(file, "cl = $cl\n")
                    
                    if cl < startClus || cl > lastClus
                        continue
                    end
                    
                    cl = cl - startClus + 1
                    @assert cl >= 1 
                    @assert cl <= MaxHitsInIter
                    
                    x = xx(digis, i)
               #     write(file, "x = $x\n")
                    
                    y = yy(digis, i)
               #     write(file, "y = $y\n")
                    

                    # write(file, "$(adc(digis,i))\n")

                    ch = min(adc(digis, i), pixmx)
                    # write(file, "ch = $ch\n")
                    
                    clusParams.charge[cl] = clusParams.charge[cl] + ch
               #     write(file, "clusParams.charge[$cl] = $(clusParams.charge[cl])\n")
                    
                    if clusParams.minRow[cl] == x
                        clusParams.Q_f_X[cl] = clusParams.Q_f_X[cl] + ch
                    end
               #     write(file, "clusParams.Q_f_X[$cl] = $(clusParams.Q_f_X[cl])\n")
                    
                    if clusParams.maxRow[cl] == x
                        clusParams.Q_l_X[cl] = clusParams.Q_l_X[cl] + ch
                    end
               #     write(file, "clusParams.Q_l_X[$cl] = $(clusParams.Q_l_X[cl])\n")
                    
                    if clusParams.minCol[cl] == y
                        clusParams.Q_f_Y[cl] = clusParams.Q_f_Y[cl] + ch
                    end
               #     write(file, "clusParams.Q_f_Y[$cl] = $(clusParams.Q_f_Y[cl])\n")
                    
                    if clusParams.maxCol[cl] == y
                        clusParams.Q_l_Y[cl] = clusParams.Q_l_Y[cl] + ch
                    end
               #     write(file, "clusParams.Q_l_Y[$cl] = $(clusParams.Q_l_Y[cl])\n")
                end

                #write(file,"###########################################\n")

                first = clus_module_start(clusters, UInt32(me + 1)) + startClus
                #write(file, "first = $first\n")

                # exit(404)
                #write(file, "FOR ic in 1 to $nClusInIter\n")
                for ic in 1:nClusInIter
                    #write(file,"########################################// $ic \n")
                    h = UInt32(first - 1 + ic)
                    #write(file,"h is: $h\n")
                    if (h > max_hits())
                        break
                    end

                    @assert h <= n_hits(hits)
                    @assert h <= clus_module_start(clusters, UInt32(me + 2))
                    # println(h)
               #     write(file,"n_hits = $(n_hits(hits))\n")
               #     write(file,"clus_module_start = $(clus_module_start(clusters, UInt32(me + 2)))\n")


                    position_corr(commonParams(cpeParams), detParams(cpeParams,UInt32(me + 1)), clusParams, UInt32(ic));
                    errorFromDB(commonParams(cpeParams), detParams(cpeParams,UInt32(me + 1)), clusParams, UInt32(ic));
                    
                    charge(hits, h, clusParams.charge[ic])
               #     write(file, "clusParams.charge[$ic] = $(clusParams.charge[ic])\n")

                    detector_index(hits, h, me)
               #     write(file, "detector_index[$h] = $(me)\n")


                    xl = x_local(hits, h, clusParams.xpos[ic])
               #     write(file, "clusParams.xpos[$ic] = $(clusParams.xpos[ic])\n")

                    yl = y_local(hits, h, clusParams.ypos[ic])
               #     write(file, "clusParams.ypos[$ic] = $(clusParams.ypos[ic])\n")

                    cluster_size_x(hits, h, clusParams.xsize[ic])
                    # write(file, "clusParams.xsize[$h] = $(clusParams.xsize[ic])\n")

                    cluster_size_y(hits, h, clusParams.ysize[ic])
                    # write(file, "clusParams.ysize[$h] = $(clusParams.ysize[ic])\n")

                    xerr_local(hits, h, clusParams.xerr[ic] * clusParams.xerr[ic])
               #     write(file, "clusParams.xerr[$ic] = $(clusParams.xerr[ic] * clusParams.xerr[ic])\n")

                    yerr_local(hits, h, clusParams.yerr[ic] * clusParams.yerr[ic])
               #     write(file, "clusParams.ysize[$ic] = $(clusParams.yerr[ic] * clusParams.yerr[ic])\n")


                    xg::Float32 = 0
                    yg::Float32 = 0 
                    zg::Float32 = 0
                    
                    frame = detParams(cpeParams, UInt32(me + 1)).frame
                    # println(xg," ", yg," ", zg)
                    xg, yg, zg = toGlobal_Special(frame, xl, yl)
               #     write(file,"bs.x = $(bs.x)\n")
                    xg = xg - bs.x
                    yg = yg - bs.y
                    zg = zg - bs.z

               #     write(file, "xg = $xg\n")
               #     write(file, "yg = $yg\n")
               #     write(file, "zg = $zg\n")
                
                    set_x_global(hits, h, xg)
                    set_y_global(hits, h, yg)
                    set_z_global(hits, h, zg) 

                    r_global(hits,h,sqrt(xg * xg + yg * yg))
                    i_phi(hits, h, unsafe_atan2s(yg, xg,7))
               #     write(file,"unsafe_atan2s($yg,$xg,7) = $(unsafe_atan2s(yg, xg,7))\n")
                end
            end

        end
     #    close(file)
end 


    
end