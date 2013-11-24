#include "MemoryBuffer.h"
#include <unittest++/UnitTest++.h>
#include <cstdlib>
#include <vector>
#include "datatype.h"

class A {
public:
  float x;
  double y;
  int *ptr;
  unsigned int i;
};

TEST(Sanity)
{
  for(int i = 0; i < 1000; i++){
    int n = rand() % 100;
    std::cout << "testing with " << n << " " 
	      << data_type_name << " elements.\n";
    gm3d::MemoryBuffer<TYPE> mb(n);
    std::vector<TYPE> v(n);
    for(int j = 0; j < n; j++){
      v[j] = j;
    }
    TYPE *ptr = mb.addr(gm3d::MemoryBuffer<TYPE>::host, 
			    gm3d::MemoryBuffer<TYPE>::write);
    for(int j = 0; j < n; j++){
      ptr[j] = v[j];
    }
    std::vector<TYPE> w(n);{
      TYPE *dev_w = mb.addr(gm3d::MemoryBuffer<TYPE>::device,
				gm3d::MemoryBuffer<TYPE>::read);
      gm3d::check(cudaMemcpy(w.data(), dev_w, 
			     n * sizeof(TYPE), cudaMemcpyDeviceToHost));
      for(int j = 0; j < n; j++){
	CHECK_EQUAL(w[j], v[j]);
      }
    }
  }
}

int main(int argc, char **argv)
{
  return UnitTest::RunAllTests();
}
