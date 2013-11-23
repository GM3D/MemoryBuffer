#ifndef MEMORY_BUFFER_H
#define MEMORY_BUFFER_H

namespace gm3d {

  template <class T> class MemoryBuffer<T> {
  private:
    T *dev_buf;
    T *host_buf;
    size_t size;
    bool host_valid;
    bool dev_valid;
    mem_space last_access;
  public:
    enum mem_space {host, device, read, write, na};
    enum access_mode {read, write};
    MemoryBuffer();
    MemoryBuffer(size_t n, mem_space space=host);
    ~MemoryBuffer();
    T *addr(mem_space space = host, access_mode mode = read);
  };

};
#endif MEMORY_BUFFER_H
