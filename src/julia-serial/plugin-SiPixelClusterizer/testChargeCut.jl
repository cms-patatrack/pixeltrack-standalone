using Test

# Mock data setup
const MAX_NUM_CLUSTERS_PER_MODULES = 10
const MAX_NUM_MODULES = 100
INV_ID = 9999
moduleStart = [0, 1, 3, 5, 7]  # Example module starts
nClustersInModule = zeros(Int, MAX_NUM_MODULES + 1)
moduleId = zeros(Int, length(moduleStart) - 1)
id = [1, 1, 1, 1, 1, 1, 1, 1]
adc = [5000, 2000, 2000, 8000, 2000, 5000, 2500, 1000]
clusterId = fill(0, 10)
numElements = length(id)

function test_cluster_charge_cut()
    # Initial values setup
    nClustersInModule .= 0
    for mod in 1:length(moduleStart) - 1
        moduleId[mod] = id[moduleStart[mod + 1]]
    end
    
    # Call the function
    gpuClusterCharge.cluster_charge_cut(id, adc, moduleStart, nClustersInModule, moduleId, clusterId, numElements)
    println(nClustersInModule)
    # Assertions
    @test all(nClustersInModule .>= 0)  # Ensure no negative values
    @test all(nClustersInModule .<= MAX_NUM_CLUSTERS_PER_MODULES)  # Ensure within bounds
    # Add more assertions based on expected behavior
end

test_cluster_charge_cut()
