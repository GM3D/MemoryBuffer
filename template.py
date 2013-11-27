from string import Template
head = '''namespace gm3d {
  void instantiate_templates(){'''

body = Template('''
    MemoryBuffer<$type> mb0_${t2}_a(0);
    MemoryBuffer<$type> mb0_${t2}_b(mb0_${t2}_a);
    std::vector<$type> v0_${t2} = mb0_${t2}_b.to_host_vector();
    MemoryBuffer<$type> mb0_${t2}_c(v0_${t2});
    MemoryBuffer<$type> mb0_${t2}_d(0);
    mb0_${t2}_d = mb0_${t2}_c;
    std::cout << mb0_${t2}_d;
    std::cout << mb0_${t2}_d.addr();
    mb0_${t2}_d.print();
    ''')

body2 = Template('''
    MemoryBuffer<$type> mb0_${t2}_a(0);
    thrust::device_vector<$type> w0_${t2} = mb0_${t2}_a.to_dev_vector();
    MemoryBuffer<$type> mb0_${t2}_b(w0_${t2});
    mb0_${t2}_b.print();
''')

foot = '''
  }
};'''

signables = ('char', 'int', 'short', 'long')
unsignables = ('float', 'double')

d = {'':'', 'signed ':'s', 'unsigned ':'u'}

f=open('MemoryBuffer_templates.cpp', 'wt')
f2=open('MemoryBuffer_templates.cu', 'wt')
f.write(head)
for t in signables + unsignables:
    if t in signables:
        for sign in ('', 'signed ', 'unsigned '):
            t1 = sign + t
            t2 = d[sign] + t
            f.write(body.substitute({'type':t1, 't2':t2}))
            f2.write(body2.substitute({'type':t1, 't2':t2}))
    else:
        t1 = t
        t2 = t
        f.write(body.substitute({'type':t1, 't2':t2}))
        f2.write(body2.substitute({'type':t1, 't2':t2}))

f.write(foot)
f.close()
f2.close()




