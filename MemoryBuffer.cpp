#include "MemoryBuffer.h"

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

  template <class T> MemoryBuffer<T>::MemoryBuffer(const std::vector<T> &v)
    : n(v.size()), host_current(false), dev_current(false),
      host_buf(NULL), dev_buf(NULL)
  {
    size_t size = n * sizeof(T);
    host_buf = new T[n];
    memcpy(host_buf, v.data(), size);
    host_current = true;
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
  
  template <class T> void MemoryBuffer<T>::
  print(const char *header, std::ostream &o)
  {
    T *hostPtr = addr(host, read);
    // (std::ostream& o, const char *header, const T *ptr, 
    //  size_t n, const size_t nMax)
    {
      std::ios state1(NULL);
      state1.copyfmt(o);
      size_t i;
      size_t omitted = 0;
      // if(n > nMax){
      //   omitted = n - nMax;
      //   n = nMax;
      // }
      // T *hostPtr = new T[n];
      // CUDACHECK(cudaMemcpy, hostPtr, ptr, sizeof(T) * n, cudaMemcpyDeviceToHost);
      o << header << ": {";
      size_t printed = 0, dupCount = 1;
      //    o << std::internal << std::showpoint;
      for(i = 1; i < n; i++){
	if(dupContinues(hostPtr[i - 1], hostPtr[i])){
	  dupCount++;
	  continue;
	}
	o << hostPtr[i - 1];
	printed++;
	if(dupCount > 1){
	  o << " <repeat " << dupCount << " times>";
	  printed += dupCount - 1;
	  dupCount = 1;
	}
	if(i < n){
	  o << ", ";
	}
      }
      if(printed < n){
	o << hostPtr[i - 1];
	printed++;
	if(dupCount > 1){
	  o << " <repeat " << dupCount << "times>";
	}
      }
      if(omitted > 0){
	o << " (" << omitted << " more elements.)";
      }
      o << "}";
      o << "\n";
      o.copyfmt(state1);
    }
  }

  template <> int MemoryBuffer<double>::
  dupContinues(const double &a, const double &b)
  {
    int a1, b1;
    a1 = std::fpclassify(a);
    b1 = std::fpclassify(b);
    if(a1 != b1) return 0;
    switch(a1){
    case FP_NAN:
    case FP_INFINITE:
      return (std::signbit(a) == std::signbit(b));
    default:
      return (a == b);
    }
  }

  template <> int MemoryBuffer<float>::
  dupContinues(const float &a, const float &b)
  {
    int a1, b1;
    a1 = std::fpclassify(a);
    b1 = std::fpclassify(b);
    if(a1 != b1) return 0;
    switch(a1){
    case FP_NAN:
    case FP_INFINITE:
      return (std::signbit(a) == std::signbit(b));
    default:
      return (a == b);
    }
  }

  template <class T> int MemoryBuffer<T>::dupContinues(const T &a, const T &b)
  {
    return (a == b);
  }

  template <class T> std::vector<T> MemoryBuffer<T>::
  to_host_vector()
  {
    std::vector<T> v(n);
    memcpy(v.data(), addr(host, read), n * sizeof(T));
    return v;
  }

  template <class T> std::ostream& 
  operator<<(std::ostream& o, const MemoryBuffer<T> &m)
  {
    o << "{n: " << m.n;
    o<< ", host_buf";
    if(m.host_current) o << "(current)";
    o << ": " << m.host_buf;
    o << ", dev_buf";
    if(m.dev_current) o << "(current)";
    o << ": " << m.dev_buf;
    o << "}";
    return o;
  }

  void instantiate_templates(){
    MemoryBuffer<char> mb0_char_a(0);
    MemoryBuffer<char> mb0_char_b(mb0_char_a);
    std::vector<char> v0_char = mb0_char_b.to_host_vector();
    MemoryBuffer<char> mb0_char_c(v0_char);
    MemoryBuffer<char> mb0_char_d(0);
    mb0_char_d = mb0_char_c;
    std::cout << mb0_char_d;
    std::cout << mb0_char_d.addr();
    mb0_char_d.print();
    
    MemoryBuffer<signed char> mb0_schar_a(0);
    MemoryBuffer<signed char> mb0_schar_b(mb0_schar_a);
    std::vector<signed char> v0_schar = mb0_schar_b.to_host_vector();
    MemoryBuffer<signed char> mb0_schar_c(v0_schar);
    MemoryBuffer<signed char> mb0_schar_d(0);
    mb0_schar_d = mb0_schar_c;
    std::cout << mb0_schar_d;
    std::cout << mb0_schar_d.addr();
    mb0_schar_d.print();
    
    MemoryBuffer<unsigned char> mb0_uchar_a(0);
    MemoryBuffer<unsigned char> mb0_uchar_b(mb0_uchar_a);
    std::vector<unsigned char> v0_uchar = mb0_uchar_b.to_host_vector();
    MemoryBuffer<unsigned char> mb0_uchar_c(v0_uchar);
    MemoryBuffer<unsigned char> mb0_uchar_d(0);
    mb0_uchar_d = mb0_uchar_c;
    std::cout << mb0_uchar_d;
    std::cout << mb0_uchar_d.addr();
    mb0_uchar_d.print();
    
    MemoryBuffer<int> mb0_int_a(0);
    MemoryBuffer<int> mb0_int_b(mb0_int_a);
    std::vector<int> v0_int = mb0_int_b.to_host_vector();
    MemoryBuffer<int> mb0_int_c(v0_int);
    MemoryBuffer<int> mb0_int_d(0);
    mb0_int_d = mb0_int_c;
    std::cout << mb0_int_d;
    std::cout << mb0_int_d.addr();
    mb0_int_d.print();
    
    MemoryBuffer<signed int> mb0_sint_a(0);
    MemoryBuffer<signed int> mb0_sint_b(mb0_sint_a);
    std::vector<signed int> v0_sint = mb0_sint_b.to_host_vector();
    MemoryBuffer<signed int> mb0_sint_c(v0_sint);
    MemoryBuffer<signed int> mb0_sint_d(0);
    mb0_sint_d = mb0_sint_c;
    std::cout << mb0_sint_d;
    std::cout << mb0_sint_d.addr();
    mb0_sint_d.print();
    
    MemoryBuffer<unsigned int> mb0_uint_a(0);
    MemoryBuffer<unsigned int> mb0_uint_b(mb0_uint_a);
    std::vector<unsigned int> v0_uint = mb0_uint_b.to_host_vector();
    MemoryBuffer<unsigned int> mb0_uint_c(v0_uint);
    MemoryBuffer<unsigned int> mb0_uint_d(0);
    mb0_uint_d = mb0_uint_c;
    std::cout << mb0_uint_d;
    std::cout << mb0_uint_d.addr();
    mb0_uint_d.print();
    
    MemoryBuffer<short> mb0_short_a(0);
    MemoryBuffer<short> mb0_short_b(mb0_short_a);
    std::vector<short> v0_short = mb0_short_b.to_host_vector();
    MemoryBuffer<short> mb0_short_c(v0_short);
    MemoryBuffer<short> mb0_short_d(0);
    mb0_short_d = mb0_short_c;
    std::cout << mb0_short_d;
    std::cout << mb0_short_d.addr();
    mb0_short_d.print();
    
    MemoryBuffer<signed short> mb0_sshort_a(0);
    MemoryBuffer<signed short> mb0_sshort_b(mb0_sshort_a);
    std::vector<signed short> v0_sshort = mb0_sshort_b.to_host_vector();
    MemoryBuffer<signed short> mb0_sshort_c(v0_sshort);
    MemoryBuffer<signed short> mb0_sshort_d(0);
    mb0_sshort_d = mb0_sshort_c;
    std::cout << mb0_sshort_d;
    std::cout << mb0_sshort_d.addr();
    mb0_sshort_d.print();
    
    MemoryBuffer<unsigned short> mb0_ushort_a(0);
    MemoryBuffer<unsigned short> mb0_ushort_b(mb0_ushort_a);
    std::vector<unsigned short> v0_ushort = mb0_ushort_b.to_host_vector();
    MemoryBuffer<unsigned short> mb0_ushort_c(v0_ushort);
    MemoryBuffer<unsigned short> mb0_ushort_d(0);
    mb0_ushort_d = mb0_ushort_c;
    std::cout << mb0_ushort_d;
    std::cout << mb0_ushort_d.addr();
    mb0_ushort_d.print();
    
    MemoryBuffer<long> mb0_long_a(0);
    MemoryBuffer<long> mb0_long_b(mb0_long_a);
    std::vector<long> v0_long = mb0_long_b.to_host_vector();
    MemoryBuffer<long> mb0_long_c(v0_long);
    MemoryBuffer<long> mb0_long_d(0);
    mb0_long_d = mb0_long_c;
    std::cout << mb0_long_d;
    std::cout << mb0_long_d.addr();
    mb0_long_d.print();
    
    MemoryBuffer<signed long> mb0_slong_a(0);
    MemoryBuffer<signed long> mb0_slong_b(mb0_slong_a);
    std::vector<signed long> v0_slong = mb0_slong_b.to_host_vector();
    MemoryBuffer<signed long> mb0_slong_c(v0_slong);
    MemoryBuffer<signed long> mb0_slong_d(0);
    mb0_slong_d = mb0_slong_c;
    std::cout << mb0_slong_d;
    std::cout << mb0_slong_d.addr();
    mb0_slong_d.print();
    
    MemoryBuffer<unsigned long> mb0_ulong_a(0);
    MemoryBuffer<unsigned long> mb0_ulong_b(mb0_ulong_a);
    std::vector<unsigned long> v0_ulong = mb0_ulong_b.to_host_vector();
    MemoryBuffer<unsigned long> mb0_ulong_c(v0_ulong);
    MemoryBuffer<unsigned long> mb0_ulong_d(0);
    mb0_ulong_d = mb0_ulong_c;
    std::cout << mb0_ulong_d;
    std::cout << mb0_ulong_d.addr();
    mb0_ulong_d.print();
    
    MemoryBuffer<float> mb0_float_a(0);
    MemoryBuffer<float> mb0_float_b(mb0_float_a);
    std::vector<float> v0_float = mb0_float_b.to_host_vector();
    MemoryBuffer<float> mb0_float_c(v0_float);
    MemoryBuffer<float> mb0_float_d(0);
    mb0_float_d = mb0_float_c;
    std::cout << mb0_float_d;
    std::cout << mb0_float_d.addr();
    mb0_float_d.print();
    
    MemoryBuffer<double> mb0_double_a(0);
    MemoryBuffer<double> mb0_double_b(mb0_double_a);
    std::vector<double> v0_double = mb0_double_b.to_host_vector();
    MemoryBuffer<double> mb0_double_c(v0_double);
    MemoryBuffer<double> mb0_double_d(0);
    mb0_double_d = mb0_double_c;
    std::cout << mb0_double_d;
    std::cout << mb0_double_d.addr();
    mb0_double_d.print();
    
  }

};




