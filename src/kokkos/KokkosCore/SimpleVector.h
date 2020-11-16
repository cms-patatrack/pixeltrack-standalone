#ifndef HeterogeneousCore_CUDAUtilities_interface_SimpleVector_h
#define HeterogeneousCore_CUDAUtilities_interface_SimpleVector_h

//  author: Felice Pantaleo, CERN, 2018

#include <type_traits>
#include <utility>

#include "KokkosCore/kokkosConfig.h"

namespace cms {
  namespace kokkos {
    template <class T>
    struct SimpleVector {
      constexpr SimpleVector() = default;

      // ownership of m_data stays within the caller
      constexpr void construct(int capacity, T *data) {
        m_size = 0;
        m_capacity = capacity;
        m_data = data;
      }

      inline constexpr int push_back_unsafe(const T &element) {
        auto previousSize = m_size;
        m_size++;
        if (previousSize < m_capacity) {
          m_data[previousSize] = element;
          return previousSize;
        } else {
          --m_size;
          return -1;
        }
      }

      template <class... Ts>
      constexpr int emplace_back_unsafe(Ts &&... args) {
        auto previousSize = m_size;
        m_size++;
        if (previousSize < m_capacity) {
          (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
          return previousSize;
        } else {
          --m_size;
          return -1;
        }
      }

      inline constexpr T &back() { return m_data[m_size - 1]; }

      inline constexpr const T &back() const {
        if (m_size > 0) {
          return m_data[m_size - 1];
        } else
          return T();  //undefined behaviour
      }

      // thread-safe version of the vector, when used in a CUDA kernel
      KOKKOS_INLINE_FUNCTION int push_back(const T &element) {
        auto previousSize = Kokkos::atomic_fetch_add(&m_size, 1);
        if (previousSize < m_capacity) {
          m_data[previousSize] = element;
          return previousSize;
        } else {
          Kokkos::atomic_sub(&m_size, 1);
          return -1;
        }
      }

      template <class... Ts>
      KOKKOS_INLINE_FUNCTION int emplace_back(Ts &&... args) {
        auto previousSize = Kokkos::atomic_fetch_add(&m_size, 1);
        if (previousSize < m_capacity) {
          (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
          return previousSize;
        } else {
          Kokkos::atomic_sub(&m_size, 1);
          return -1;
        }
      }

      // thread safe version of resize
      KOKKOS_INLINE_FUNCTION int extend(int size = 1) {
        auto previousSize = Kokkos::atomic_fetch_add(&m_size, size);
        if (previousSize < m_capacity) {
          return previousSize;
        } else {
          Kokkos::atomic_sub(&m_size, size);
          return -1;
        }
      }

      KOKKOS_INLINE_FUNCTION int shrink(int size = 1) {
        auto previousSize = Kokkos::atomic_fetch_sub(&m_size, size);
        if (previousSize >= size) {
          return previousSize - size;
        } else {
          Kokkos::atomic_add(&m_size, size);
          return -1;
        }
      }

      inline constexpr bool empty() const { return m_size <= 0; }
      inline constexpr bool full() const { return m_size >= m_capacity; }
      inline constexpr T &operator[](int i) { return m_data[i]; }
      inline constexpr const T &operator[](int i) const { return m_data[i]; }
      inline constexpr void reset() { m_size = 0; }
      inline constexpr int size() const { return m_size; }
      inline constexpr int capacity() const { return m_capacity; }
      inline constexpr T const *data() const { return m_data; }
      inline constexpr void resize(int size) { m_size = size; }
      inline constexpr void set_data(T *data) { m_data = data; }

    private:
      int m_size;
      int m_capacity;

      T *m_data;
    };

    // ownership of m_data stays within the caller
    template <class T>
    SimpleVector<T> make_SimpleVector(int capacity, T *data) {
      SimpleVector<T> ret;
      ret.construct(capacity, data);
      return ret;
    }

    // ownership of m_data stays within the caller
    template <class T>
    SimpleVector<T> *make_SimpleVector(SimpleVector<T> *mem, int capacity, T *data) {
      auto ret = new (mem) SimpleVector<T>();
      ret->construct(capacity, data);
      return ret;
    }
  }  // namespace kokkos
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_GPUSimpleVector_h
