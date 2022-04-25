// C++ headers
#include <algorithm>
#include <numeric>

// CMSSW headers
#include "CUDACore/cudaCompat.h"

#include "plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.h"  // !
#include "plugin-SiPixelClusterizer/gpuClusteringConstants.h"        // !

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>


std::string path = "/home/simone/Documents/thesis/not_sorted/";
int n_events = 1770;

std::map<int,int> def_hits_map() {
  std::map<int,int> event_nhits = {};
  std::ifstream is_;
  is_.open(path + "hits_per_event.csv");
  std::string a;

  for(int i = 0; is_ >> a; ++i) {
    std::stringstream str(a);
    std::string row;
    std::vector<int> key_values;
  
    while(std::getline(str, row, ',')) {
      key_values.push_back(std::stoi(row));
    }
    event_nhits[key_values[0]] = key_values[1];
  }
  return event_nhits;
}

std::map<int,int> n_hits_map = def_hits_map();

namespace {
  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                    uint32_t* hitsLayerStart) {
    assert(0 == hitsModuleStart[0]);

    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int end = 11;
    for (int i = begin; i < end; i += blockDim.x * gridDim.x) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

  // crea un nuovo costruttore con i miei vettori
  TrackingRecHit2DCPU PixelRecHitGPUKernel::makeHits(SiPixelDigisSoA const& digis_d,
                                                     SiPixelClustersSoA const& clusters_d,
                                                     BeamSpotPOD const& bs_d,
                                                     pixelCPEforGPU::ParamsOnGPU const* cpeParams) const {
    auto nHits = clusters_d.nClusters();  // size vettori
    TrackingRecHit2DCPU hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), nullptr);  // costruttore

    if (digis_d.nModules())  // protect from empty events
      gpuPixelRecHits::getHits(cpeParams, &bs_d, digis_d.view(), digis_d.nDigis(), clusters_d.view(), hits_d.view());

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      setHitsLayerStart(clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart());
    }

    if (nHits) {
      cms::cuda::fillManyFromVector(hits_d.phiBinner(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256);
    }

    return hits_d;
  }

  TrackingRecHit2DCPU PixelRecHitGPUKernel::makeHits(int file_number) const {
    std::vector<double> hits_x_coordinates;
    std::vector<double> hits_y_coordinates;
    std::vector<double> hits_z_coordinates;
    std::vector<double> hits_r_coordinates;
  
    if(file_number >= 5000 && file_number < 5500) {
      std::cout << "This file is missing" << '\n';
    } else {
    std::string x_file_name = path + "x_ns" + std::to_string(file_number) + ".dat";
    std::string y_file_name = path + "y_ns" + std::to_string(file_number) + ".dat";
    std::string z_file_name = path + "z_ns" + std::to_string(file_number) + ".dat";
    
    // Read the x_ns*.dat.dat file
    std::ifstream is_1, is_2, is_3;
    is_1.open(x_file_name);
    int a;

    // Create the vector containing all the x coordinates of the hits
    for(int i = 0; is_1 >> a; ++i) { hits_x_coordinates.push_back(a); }
    is_1.close();

    // Read the y_ns*.dat.dat file
    is_2.open(y_file_name);
    int b;

    // Create the vector containing all the y coordinates of the hits
    for(int i = 0; is_2 >> b; ++i) { hits_y_coordinates.push_back(b); }
    is_2.close();

    // Read the z_ns*.dat.dat file
    is_3.open(z_file_name);
    int c;

    // Create the vector containing all the z coordinates of the hits
    for(int i = 0; is_3 >> c; ++i) { hits_z_coordinates.push_back(c); }
    is_3.close();

    for(int i = 0 ; i < static_cast<int>(hits_y_coordinates.size()); ++i) {
      hits_r_coordinates.push_back(sqrt(pow(hits_y_coordinates[i],2) + pow(hits_z_coordinates[i],2)));
    }

    std::cout << hits_z_coordinates.size() << '\n';
    std::cout << n_hits_map[file_number] << '\n';
  }

    TrackingRecHit2DCPU hits_d(hits_x_coordinates, hits_y_coordinates, hits_z_coordinates, hits_r_coordinates);
    return hits_d;
  }

}  // namespace pixelgpudetails
