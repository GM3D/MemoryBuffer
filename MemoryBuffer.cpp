namespace gm3d{

  template <class T> MemoryBuffer<T>::MemoryBuffer()
    : dev_buf(NULL), host_buf(NULL), n(0), 
      host_current(false), dev_current(false)
  {
  }

  template <class T> MemoryBuffer<T>::MemoryBuffer(size_t n0, mem_space space)
    : dev_buf(NULL), host_buf(NULL), n(n0), 
      host_current(false), dev_current(false)
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

  template <class T> MemoryBuffer<T>::~MemoryBuffer()
  {
    delete[] host_buf;
    check(cudaFree(dev_buf));
  }
  
  template <class T> T *MemoryBuffer<T>::
  addr(mem_space space, access_mode mode) throw(std::exception)
  {
    if(space == host){
      /// host memory requested. allocate if it isn't.
      if(host_buf == NULL){
	host_buf = new T[n];
      }
      if(mode == write){
	/// write host. host becomes current unconditionally.
	host_current = true;
	dev_current = false;
      }else{
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
	  }else{
	    /// neither of host or dev is current, this should not happen.
	    throw std::domain_error("memory space neither host nor device");
	  }
	} // end host_current
      }
      return host_buf;
      // end host memory space requested.
    }else if(space == device){
      if(dev_buf == NULL){
	check(cudaMalloc(&dev_buf, n * sizeof(T)));
      }
      if(mode == write){
	dev_current = true;
	host_current = false;
      }else{
	/// read mode.
	if(dev_current){
	  /// just make sure buf is allocated.
	  assert(dev_buf);
	}else{
	  /// dev not current, check if host is current.
	  if(host_current){
	    check(cudaMemcpy(dev_buf, host_buf, n * sizeof(T), 
			     cudaMemcpyHostToDevice));
	    dev_current = true;
	  }else{
	    /// neither of host or dev is current, this should not happen.
	    throw std::domain_error("neither of host or device is current");
	  }
	}// end dev_current
      }
      return dev_buf;
      // end device memory space requested.
    }else{
      /// memory space neither host or device requested.
      throw std::invalid_argument("invalid memory space requested.");
    }
  }
}
