#ifndef MEMORY_BUFFER_H
#define MEMORY_BUFFER_H
#include "cudaUtils.h"
#include <iostream>
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <vector>
#include <thrust/device_vector.h>

namespace gm3d {

  template <class T> class MemoryBuffer {
    /** Abstraction class to handle CUDA host and device memory. */
  private:
    /// n is the number of element, not bytes.
    size_t n;
    bool host_current;
    bool dev_current;
    T *host_buf;
    T *dev_buf;
    /** semantics;
	if host_current is true, host_buf is valid non-NULL host address.
	if device_current is true, device_buf is valid non-NULL device address.
	host_current and device_current is not mutually exclusive.
	if both are true, their content must coincide.
    */
    /// for intenal use (in print())
    int dupContinues(const T &a, const T&b);
  public:
    enum mem_space {host, device, na};
    enum access_mode {read, write};
    MemoryBuffer(size_t n, mem_space space = host);
    MemoryBuffer(const MemoryBuffer &m);
    MemoryBuffer(const std::vector<T> &v);
    MemoryBuffer(const thrust::device_vector<T> &v);
    ~MemoryBuffer();
    MemoryBuffer& operator=(const MemoryBuffer& m);
    T *addr(mem_space space = host, access_mode mode = read)
      throw(std::exception);
    void print(const char *header=NULL, std::ostream &o=std::cout);
    std::vector<T> to_host_vector();
    thrust::device_vector<T> to_dev_vector();
    template <class T2> friend std::ostream& 
      operator<<(std::ostream& o, const MemoryBuffer<T2> &m);
  };

};

#endif // MEMORY_BUFFER_H
