
    MemoryBuffer<char> mb0_char_a(0);
    thrust::device_vector<char> w0_char = mb0_char_a.to_dev_vector();
    MemoryBuffer<char> mb0_char_b(w0_char);
    mb0_char_b.print();

    MemoryBuffer<signed char> mb0_schar_a(0);
    thrust::device_vector<signed char> w0_schar = mb0_schar_a.to_dev_vector();
    MemoryBuffer<signed char> mb0_schar_b(w0_schar);
    mb0_schar_b.print();

    MemoryBuffer<unsigned char> mb0_uchar_a(0);
    thrust::device_vector<unsigned char> w0_uchar = mb0_uchar_a.to_dev_vector();
    MemoryBuffer<unsigned char> mb0_uchar_b(w0_uchar);
    mb0_uchar_b.print();

    MemoryBuffer<int> mb0_int_a(0);
    thrust::device_vector<int> w0_int = mb0_int_a.to_dev_vector();
    MemoryBuffer<int> mb0_int_b(w0_int);
    mb0_int_b.print();

    MemoryBuffer<signed int> mb0_sint_a(0);
    thrust::device_vector<signed int> w0_sint = mb0_sint_a.to_dev_vector();
    MemoryBuffer<signed int> mb0_sint_b(w0_sint);
    mb0_sint_b.print();

    MemoryBuffer<unsigned int> mb0_uint_a(0);
    thrust::device_vector<unsigned int> w0_uint = mb0_uint_a.to_dev_vector();
    MemoryBuffer<unsigned int> mb0_uint_b(w0_uint);
    mb0_uint_b.print();

    MemoryBuffer<short> mb0_short_a(0);
    thrust::device_vector<short> w0_short = mb0_short_a.to_dev_vector();
    MemoryBuffer<short> mb0_short_b(w0_short);
    mb0_short_b.print();

    MemoryBuffer<signed short> mb0_sshort_a(0);
    thrust::device_vector<signed short> w0_sshort = mb0_sshort_a.to_dev_vector();
    MemoryBuffer<signed short> mb0_sshort_b(w0_sshort);
    mb0_sshort_b.print();

    MemoryBuffer<unsigned short> mb0_ushort_a(0);
    thrust::device_vector<unsigned short> w0_ushort = mb0_ushort_a.to_dev_vector();
    MemoryBuffer<unsigned short> mb0_ushort_b(w0_ushort);
    mb0_ushort_b.print();

    MemoryBuffer<long> mb0_long_a(0);
    thrust::device_vector<long> w0_long = mb0_long_a.to_dev_vector();
    MemoryBuffer<long> mb0_long_b(w0_long);
    mb0_long_b.print();

    MemoryBuffer<signed long> mb0_slong_a(0);
    thrust::device_vector<signed long> w0_slong = mb0_slong_a.to_dev_vector();
    MemoryBuffer<signed long> mb0_slong_b(w0_slong);
    mb0_slong_b.print();

    MemoryBuffer<unsigned long> mb0_ulong_a(0);
    thrust::device_vector<unsigned long> w0_ulong = mb0_ulong_a.to_dev_vector();
    MemoryBuffer<unsigned long> mb0_ulong_b(w0_ulong);
    mb0_ulong_b.print();

    MemoryBuffer<float> mb0_float_a(0);
    thrust::device_vector<float> w0_float = mb0_float_a.to_dev_vector();
    MemoryBuffer<float> mb0_float_b(w0_float);
    mb0_float_b.print();

    MemoryBuffer<double> mb0_double_a(0);
    thrust::device_vector<double> w0_double = mb0_double_a.to_dev_vector();
    MemoryBuffer<double> mb0_double_b(w0_double);
    mb0_double_b.print();
