#ifndef HeterogeneousCore_CUDAUtilities_interface_SimpleVector_h
#define HeterogeneousCore_CUDAUtilities_interface_SimpleVector_h

//  author: Felice Pantaleo, CERN, 2018

#include <atomic>
#include <type_traits>
#include <utility>

namespace cms {
  namespace cuda {

    template <class T>
    struct SimpleVector {
      constexpr SimpleVector() = default;

      // ownership of m_data stays within the caller
      constexpr void construct(int32_t capacity, T *data) {
        m_size.store(0);
        m_capacity = capacity;
        m_data = data;
      }

      inline constexpr int push_back_unsafe(const T &element) {
        auto previousSize = size();
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
      constexpr int emplace_back_unsafe(Ts &&...args) {
        auto previousSize = size();
        m_size++;
        if (previousSize < m_capacity) {
          (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
          return previousSize;
        } else {
          --m_size;
          return -1;
        }
      }

      inline T &back() { return m_data[size() - 1]; }

      inline const T &back() const {
        if (size() > 0) {
          return m_data[size() - 1];
        } else
          return T();  //undefined behaviour
      }

      // thread-safe version of the vector, when used in a CUDA kernel
      int push_back(const T &element) {
        auto previousSize = m_size.fetch_add(1);
        if (previousSize < m_capacity) {
          m_data[previousSize] = element;
          return previousSize;
        } else {
          m_size -= 1;
          return -1;
        }
      }

      template <class... Ts>
      int emplace_back(Ts &&...args) {
        auto previousSize = m_size.fetch_add(1);
        if (previousSize < m_capacity) {
          (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
          return previousSize;
        } else {
          m_size -= 1;
          return -1;
        }
      }

      // thread safe version of resize
      int extend(int32_t size = 1) {
        auto previousSize = m_size.fetch_add(size);
        if (previousSize < m_capacity) {
          return previousSize;
        } else {
          m_size -= size;
          return -1;
        }
      }

      int shrink(int32_t size = 1) {
        auto previousSize = m_size.fetch_sub(size);
        if (previousSize >= size) {
          return previousSize - size;
        } else {
          m_size += size;
          return -1;
        }
      }

      inline constexpr bool empty() const { return size() <= 0; }
      inline constexpr bool full() const { return size() >= m_capacity; }
      inline constexpr T &operator[](int i) { return m_data[i]; }
      inline constexpr const T &operator[](int i) const { return m_data[i]; }
      inline constexpr void reset() { resize(0); }
      inline constexpr std::atomic_int32_t::value_type size() const { return m_size.load(); }
      inline constexpr int32_t capacity() const { return m_capacity; }
      inline constexpr T const *data() const { return m_data; }
      inline constexpr void resize(int32_t size) { m_size.store(size); }
      inline constexpr void set_data(T *data) { m_data = data; }

    private:
      int32_t m_capacity;
      std::atomic_int32_t m_size;

      T *m_data;
    };

    // ownership of m_data stays within the caller
    template <class T>
    SimpleVector<T> make_SimpleVector(int32_t capacity, T *data) {
      SimpleVector<T> ret;
      ret.construct(capacity, data);
      return ret;
    }

    // ownership of m_data stays within the caller
    template <class T>
    SimpleVector<T> *make_SimpleVector(SimpleVector<T> *mem, int32_t capacity, T *data) {
      auto ret = new (mem) SimpleVector<T>();
      ret->construct(capacity, data);
      return ret;
    }

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_SimpleVector_h
