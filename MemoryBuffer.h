#ifndef MEMORY_BUFFER_H
#define MEMORY_BUFFER_H
#include "cudaUtils.h"
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cassert>

namespace gm3d {

  template <class T> class MemoryBuffer {
    /** Abstraction class to handle CUDA host and device memory. */
  private:
    T *dev_buf;
    T *host_buf;
    /// n is the number of element, not bytes.
    size_t n;
    bool host_current;
    bool dev_current;
    /** semantics;
	if host_current is true, host_buf is valid non-NULL host address.
	if device_current is true, device_buf is valid non-NULL device address.
	host_current and device_current is not mutually exclusive.
	if both are true, their content must coincide.
    */
  public:
    enum mem_space {host, device, na};
    enum access_mode {read, write};
    MemoryBuffer(size_t n = 0, mem_space space = host);
    ~MemoryBuffer();
    T *addr(mem_space space = host, access_mode mode = read)
      throw(std::exception);
  };

};

#include "MemoryBuffer.cpp"

#endif // MEMORY_BUFFER_H
