namespace gm3d {
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