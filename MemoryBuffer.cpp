namespace gm3d{

  template <class T> MemoryBuffer<T>::MemoryBuffer(size_t n0, mem_space space)
    : n(n0), host_current(false), dev_current(false),
      host_buf(NULL), dev_buf(NULL)
  {
    size_t size = n * sizeof(T);
    if(space == host){
      // allocate host memory.
      host_buf = new T[n];
      host_current = true;
    }else if(space == device){
      // allocate device memory.
      check(cudaMalloc(&dev_buf, size));
      dev_current = true;
    }
  }

  template <class T> MemoryBuffer<T>::MemoryBuffer(const MemoryBuffer<T> &m)
    : n(m.n), host_current(m.host_current), dev_current(m.dev_current),
      host_buf(NULL), dev_buf(NULL)
  {
    size_t size = n * sizeof(T);
    if(host_current){
      host_buf = new T[n];
      memcpy(host_buf, m.host_buf, size);
    }
    if(dev_current){
      check(cudaMalloc(&dev_buf, size));
      check(cudaMemcpy(dev_buf, m.dev_buf, size, cudaMemcpyDeviceToDevice));
    }
  }

  template <class T> MemoryBuffer<T>& MemoryBuffer<T>::
  operator=(const MemoryBuffer<T> &m)
  {
    if(host_buf) delete[] host_buf;
    if(dev_buf) check(cudaFree(dev_buf));
    n = m.n;
    host_current = m.host_current;
    dev_current = m.dev_current;
    host_buf = dev_buf = NULL;

    size_t size = n * sizeof(T);
    if(host_current){
      host_buf = new T[n];
      memcpy(host_buf, m.host_buf, size);
    }
    if(dev_current){
      check(cudaMalloc(&dev_buf, size));
      check(cudaMemcpy(dev_buf, m.dev_buf, size, cudaMemcpyDeviceToDevice));
    }
    return *this;
  }

  template <class T> MemoryBuffer<T>::~MemoryBuffer()
  {
    if(host_buf) delete[] host_buf;
    if(dev_buf) check(cudaFree(dev_buf));
  }
  
  template <class T> T *MemoryBuffer<T>::
  addr(mem_space space, access_mode mode) throw(std::exception)
  {
    assert(space == host || space == device);
    assert(mode == read || mode == write);
    assert(host_current || dev_current);
    if(space == host){
      /// host memory requested. allocate if it isn't.
      if(host_buf == NULL){
	host_buf = new T[n];
      }
      if(mode == write){
	/// write host. host becomes current unconditionally.
	host_current = true;
	dev_current = false;
      }else if(mode == read){
	/// read mode.
	if(host_current){
	  /// just make sure buf is allocated.
	  assert(host_buf);
	}else{
	  /// host not current, check if dev is current.
	  if(dev_current){
	    /// dev is current, copy content to host.
	    check(cudaMemcpy(host_buf, dev_buf, 
			     n * sizeof(T), cudaMemcpyDeviceToHost));
	    host_current = true;
	  }
	}// end host_current
      }else{
	throw std::invalid_argument("invalid access mode requested.");
      }
      return host_buf;
      // end host memory space requested.
    }else if(space == device){
      if(dev_buf == NULL){
	check(cudaMalloc(&dev_buf, n * sizeof(T)));
      }
      if(mode == write){
	/// device write
	dev_current = true;
	host_current = false;
      }else if(mode == read){
	/// device read
	if(dev_current){
	  /// just make sure buf is allocated before returning it.
	  assert(dev_buf);
	}else if(host_current){
	  /// dev not current, check if host is current.
	  check(cudaMemcpy(dev_buf, host_buf, n * sizeof(T), 
			   cudaMemcpyHostToDevice));
	  dev_current = true;
	}else{
	  /// this should never happen, even if arguments are invalid.
	  throw std::domain_error("neither of host or device is current");
	}
      }else{
	throw std::invalid_argument("invalid access mode requested.");
      }
      return dev_buf;
      // end device memory space requested.
    }else{
      throw std::invalid_argument("invalid memory space requested.");
    }
  }
};
