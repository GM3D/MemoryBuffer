#include <unittest++/UnitTest++.h>
#include <cstdlib>
#include <vector>
#include <thrust/device_vector.h>
#include <ctime>
#include <cstring>
#include <cmath>
#include "datatype.h"
#include "MemoryBuffer.h"

TEST(Vectors)
{
  srand(time(NULL));
  int n = rand() % 100;

  std::vector<TYPE> v0(n);
  for(int i = 0; i < n; i++){
    v0[i] = (TYPE)rand();
  }

  gm3d::MemoryBuffer<TYPE> mb0(v0);
  thrust::device_vector<TYPE> w0 = mb0.to_dev_vector();
  gm3d::MemoryBuffer<TYPE> mb1(w0);
  gm3d::MemoryBuffer<TYPE> mb2(n);
  mb2 = mb1;
  std::vector<TYPE> v1 = mb1.to_host_vector();

  for(int i = 0; i < n; i++){
    CHECK_EQUAL(v0[i], v1[i]);
  }
  mb1.print("mb1");
}


int main(int argc, char **argv)
{
  return UnitTest::RunAllTests();
}
