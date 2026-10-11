struct GPUClusteringConstants:
    # data at pileup 50 has 18300 +/- 3500 hits; 40000 is around 6 sigma away
    comptime maxNumberOfHits: UInt32 = 48 * 1024

    @staticmethod
    @always_inline
    def maxHitsInIter() -> UInt32:
        """Optimized for real data PU 50."""
        return 160

    @staticmethod
    @always_inline
    def maxHitsInModule() -> UInt32:
        return 1024

    comptime MaxNumModules: UInt32 = 2000
    comptime MaxNumClustersPerModules: Int32 = Self.maxHitsInModule().cast[
        DType.int32
    ]()
    comptime MaxHitsInModule: UInt32 = Self.maxHitsInModule()
    comptime MaxNumClusters: UInt32 = Self.maxNumberOfHits
    comptime InvId: UInt16 = 9999  # must be > MaxNumModules
