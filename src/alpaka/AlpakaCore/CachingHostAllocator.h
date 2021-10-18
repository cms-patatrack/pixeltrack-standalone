#ifndef HeterogenousCore_AlpakaUtilities_src_CachingHostAllocator_h
#define HeterogenousCore_AlpakaUtilities_src_CachingHostAllocator_h

/******************************************************************************
 * Simple caching allocator for pinned host memory allocations. The allocator is
 * thread-safe.
 ******************************************************************************/

#include <cmath>
#include <memory>
#include <unordered_set>
#include <mutex>

#include "AlpakaCore/alpakaMemoryHelper.h"
#include "AlpakaCore/deviceAllocatorStatus.h"

/// cms::alpaka::allocator namespace
namespace cms::alpakatools::allocator {

  /**
 * \addtogroup UtilMgmt
 * @{
 */

  /******************************************************************************
 * CachingHostAllocator (host use)
 ******************************************************************************/

  /**
 * \brief A simple caching allocator pinned host memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe.  It behaves as follows:
 *
 * \par
 * - Allocations are categorized and cached by bin size.  A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused host allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations  will exceed
 *   \p max_cached_bytes, allocations are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingHostAllocator is configured with:
 * - \p bin_growth          = 8
 * - \p min_bin             = 3
 * - \p max_bin             = 7
 * - \p max_cached_bytes    = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes
 *
 */

  struct CachingHostAllocator {
    //---------------------------------------------------------------------
    // Constants
    //---------------------------------------------------------------------

    /// Out-of-bounds bin
    static const unsigned int INVALID_BIN = (unsigned int)-1;

    /// Invalid size
    static const size_t INVALID_SIZE = (size_t)-1;

#ifndef DOXYGEN_SHOULD_SKIP_THIS  // Do not document

    //---------------------------------------------------------------------
    // Type definitions and helper types
    //---------------------------------------------------------------------

    /**
     * Descriptor for pinned host memory allocations
     */
    struct BlockDescriptor {
      alpaka_common::AlpakaHostBuf<std::byte> buf;  // Host buffer
      size_t bytes;                                 // Size of allocation in bytes
      unsigned int bin;                             // Bin enumeration

      // Constructor (suitable for searching maps for a specific block, given a host buffer)
      BlockDescriptor(alpaka_common::AlpakaHostBuf<std::byte> buffer)
          : buf{std::move(buffer)}, bytes{0}, bin{INVALID_BIN} {}

      // Constructor (suitable for searching maps for a block, given the bytes)
      BlockDescriptor(unsigned int block_bin, size_t block_bytes)
          : buf{allocHostBuf<std::byte>(0u)}, bytes{block_bytes}, bin{block_bin} {}
    };

    struct BlockHashByBytes {
      size_t operator()(const BlockDescriptor& descriptor) const { return std::hash<size_t>{}(descriptor.bytes); }
    };

    struct BlockEqualByBytes {
      bool operator()(const BlockDescriptor& a, const BlockDescriptor& b) const { return (a.bytes == b.bytes); }
    };

    struct BlockHashByPtr {
      size_t operator()(const BlockDescriptor& descriptor) const {
        return std::hash<const std::byte*>{}(alpaka::getPtrNative(descriptor.buf));
      }
    };

    struct BlockEqualByPtr {
      bool operator()(const BlockDescriptor& a, const BlockDescriptor& b) const {
        return (alpaka::getPtrNative(a.buf) == alpaka::getPtrNative(b.buf));
      }
    };

    /// Set type for cached blocks (hashed by size)
    using CachedBlocks = std::unordered_multiset<BlockDescriptor, BlockHashByBytes, BlockEqualByBytes>;

    /// Set type for live blocks (hashed by ptr)
    using BusyBlocks = std::unordered_multiset<BlockDescriptor, BlockHashByPtr, BlockEqualByPtr>;

    //---------------------------------------------------------------------
    // Utility functions
    //---------------------------------------------------------------------

    /**
     * Integer pow function for unsigned base and exponent
     */
    static unsigned int IntPow(unsigned int base, unsigned int exp) {
      unsigned int retval = 1;
      while (exp > 0) {
        if (exp & 1) {
          retval = retval * base;  // multiply the result by the current base
        }
        base = base * base;  // square the base
        exp = exp >> 1;      // divide the exponent in half
      }
      return retval;
    }

    /**
     * Round up to the nearest power-of
     */
    std::pair<unsigned int, size_t> NearestPowerOf(unsigned int base, size_t value) {
      unsigned int power = 0;
      size_t rounded_bytes = 1;

      if (value * base < value) {
        // Overflow
        power = sizeof(size_t) * 8;
        rounded_bytes = size_t(0) - 1;
      } else {
        while (rounded_bytes < value) {
          rounded_bytes *= base;
          power++;
        }
      }

      return {power, rounded_bytes};
    }

    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    std::mutex mutex;  /// Mutex for thread-safety

    unsigned int bin_growth;  /// Geometric growth factor for bin-sizes
    unsigned int min_bin;     /// Minimum bin enumeration
    unsigned int max_bin;     /// Maximum bin enumeration

    size_t min_bin_bytes;     /// Minimum bin size
    size_t max_bin_bytes;     /// Maximum bin size
    size_t max_cached_bytes;  /// Maximum aggregate cached bytes

    bool debug;  /// Whether or not to print (de)allocation events to stdout

    TotalBytes cached_bytes;     /// Aggregate cached bytes
    CachedBlocks cached_blocks;  /// Set of cached pinned host allocations available for reuse
    BusyBlocks live_blocks;      /// Set of live pinned host allocations currently in use

#endif  // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * \brief Constructor.
     */
    CachingHostAllocator(
        unsigned int bin_growth,                 ///< Geometric growth factor for bin-sizes
        unsigned int min_bin = 1,                ///< Minimum bin (default is bin_growth ^ 1)
        unsigned int max_bin = INVALID_BIN,      ///< Maximum bin (default is no max bin)
        size_t max_cached_bytes = INVALID_SIZE,  ///< Maximum aggregate cached bytes (default is no limit)
        bool debug = false)  ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)
        : bin_growth(bin_growth),
          min_bin(min_bin),
          max_bin(max_bin),
          min_bin_bytes(IntPow(bin_growth, min_bin)),
          max_bin_bytes(IntPow(bin_growth, max_bin)),
          max_cached_bytes(max_cached_bytes),
          debug(debug) {}

    /**
     * \brief Default constructor.
     *
     * Configured with:
     * \par
     * - \p bin_growth          = 8
     * - \p min_bin             = 3
     * - \p max_bin             = 7
     * - \p max_cached_bytes    = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
     *
     * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
     * sets a maximum of 6,291,455 cached bytes
     */
    CachingHostAllocator(bool debug = false)
        : bin_growth(8),
          min_bin(3),
          max_bin(7),
          min_bin_bytes(IntPow(bin_growth, min_bin)),
          max_bin_bytes(IntPow(bin_growth, max_bin)),
          max_cached_bytes((max_bin_bytes * 3) - 1),
          debug(debug) {}

    /**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache
     *
     * Changing the ceiling of cached bytes does not cause any allocations (in-use or
     * cached-in-reserve) to be freed.  See \p FreeAllCached().
     */
    void SetMaxCachedBytes(size_t max_cached_bytes) {
      // Lock
      std::unique_lock mutex_locker(mutex);

      if (debug) {
        printf("Changing max_cached_bytes (%lld -> %lld)\n",
               (long long)this->max_cached_bytes,
               (long long)max_cached_bytes);
      }

      this->max_cached_bytes = max_cached_bytes;

      // Unlock (redundant, kept for style uniformity)
      mutex_locker.unlock();
    }

    /**
     * \brief Provides a suitable allocation of pinned host memory for the given size.
     *
     * Once freed, the allocation becomes available immediately for reuse.
     */
    auto HostAllocate(size_t bytes  ///< [in] Minimum no. of bytes for the allocation
    ) {
      std::unique_lock<std::mutex> mutex_locker(mutex, std::defer_lock);

      // Create a block descriptor for the requested allocation
      bool found = false;
      auto [bin, bin_bytes] = NearestPowerOf(bin_growth, bytes);
      BlockDescriptor search_key{bin, bin_bytes};

      if (search_key.bin > max_bin) {
        // Bin is greater than our maximum bin: allocate the request
        // exactly and give out-of-bounds bin.  It will not be cached
        // for reuse when returned.
        search_key.bin = INVALID_BIN;
        search_key.bytes = bytes;
      } else {
        // Search for a suitable cached allocation: lock
        mutex_locker.lock();

        if (search_key.bin < min_bin) {
          // Bin is less than minimum bin: round up
          search_key.bin = min_bin;
          search_key.bytes = min_bin_bytes;
        }

        // Find a cached block in the same bin
        auto block_itr = cached_blocks.find(search_key);
        if (block_itr != cached_blocks.end()) {
          // Reuse existing cache block.  Insert into live blocks.
          found = true;
          search_key = *block_itr;

          live_blocks.insert(search_key);

          // Remove from free blocks
          cached_bytes.free -= search_key.bytes;
          cached_bytes.live += search_key.bytes;

          if (debug) {
            printf("\tHost reused cached block at %p (%lld bytes).\n",
                   alpaka::getPtrNative(search_key.buf),
                   (long long)search_key.bytes);
          }

          cached_blocks.erase(block_itr);
        }

        // Done searching: unlock
        mutex_locker.unlock();
      }

      // Allocate the block if necessary
      if (!found) {
        // TODO: eventually support allocation flags
        search_key.buf = allocHostBuf<std::byte>(static_cast<alpaka_common::Extent>(search_key.bytes));
#if CUDA_VERSION >= 11020
        alpaka::prepareForAsyncCopy(search_key.buf);
#endif

        // Insert into live blocks
        mutex_locker.lock();
        live_blocks.insert(search_key);
        cached_bytes.live += search_key.bytes;
        mutex_locker.unlock();

        if (debug) {
          printf("\tHost allocated new host block at %p (%lld bytes).\n",
                 alpaka::getPtrNative(search_key.buf),
                 (long long)search_key.bytes);
        }
      }

      if (debug) {
        printf("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
               (long long)cached_blocks.size(),
               (long long)cached_bytes.free,
               (long long)live_blocks.size(),
               (long long)cached_bytes.live);
      }

      return search_key.buf;
    }

    /**
     * \brief Frees a live allocation of pinned host memory, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse.
     */
    void HostFree(const alpaka_common::AlpakaHostBuf<std::byte>& buf) {
      // Lock
      std::unique_lock<std::mutex> mutex_locker(mutex);

      bool recached = false;
      // Find corresponding block descriptor
      BlockDescriptor search_key{buf};
      auto block_itr = live_blocks.find(search_key);
      if (block_itr != live_blocks.end()) {
        // Remove from live blocks
        search_key = *block_itr;
        live_blocks.erase(block_itr);
        cached_bytes.live -= search_key.bytes;

        // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
        if ((search_key.bin != INVALID_BIN) && (cached_bytes.free + search_key.bytes <= max_cached_bytes)) {
          recached = true;
          // Insert returned allocation into free blocks
          cached_blocks.insert(search_key);
          cached_bytes.free += search_key.bytes;

          if (debug) {
            printf(
                "\tHost returned %lld bytes.\n\t\t %lld "
                "available blocks cached (%lld bytes), %lld live blocks outstanding. (%lld bytes)\n",
                (long long)search_key.bytes,
                (long long)cached_blocks.size(),
                (long long)cached_bytes.free,
                (long long)live_blocks.size(),
                (long long)cached_bytes.live);
          }
        }
      }

      // Unlock
      mutex_locker.unlock();

      if (!recached and debug) {
        printf(
            "\tHost freed %lld bytes.\n\t\t  %lld available "
            "blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
            (long long)search_key.bytes,
            (long long)cached_blocks.size(),
            (long long)cached_bytes.free,
            (long long)live_blocks.size(),
            (long long)cached_bytes.live);
      }
    }
  };

  /** @} */  // end group UtilMgmt

}  // namespace cms::alpakatools::allocator

#endif
